from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ppi_ddg.constants import AA_TO_INDEX
from ppi_ddg.data.mutations import parse_mutation_list

__all__ = ["audit_paired_ddg_cache"]


_CHECK_NAMES = (
    "mt_pdb_exists",
    "mutation_aa_differs",
    "wt_mt_coordinates_differ",
    "mutation_masks_present",
)


def _empty_report(num_samples: int) -> dict[str, Any]:
    return {
        "num_samples": int(num_samples),
        "num_failures": 0,
        "checks": {name: {"failures": 0} for name in _CHECK_NAMES},
        "examples": [],
    }


def _sample_id(item: dict[str, Any], path: Path) -> str:
    return str(item.get("sample_id") or item.get("row_id") or path.stem)


def _add_failure(
    report: dict[str, Any],
    failed_samples: set[str],
    *,
    check: str,
    item: dict[str, Any],
    path: Path,
    reason: str,
    max_examples: int,
) -> None:
    sample_id = _sample_id(item, path)
    failed_samples.add(sample_id)
    report["checks"][check]["failures"] += 1
    if len(report["examples"]) < max_examples:
        report["examples"].append(
            {
                "check": check,
                "sample_id": sample_id,
                "row_id": str(item.get("row_id", "")),
                "pdb_id": str(item.get("pdb_id", "")),
                "mutation": str(item.get("mutation", "")),
                "path": str(path),
                "reason": reason,
            }
        )


def _residue_index_by_key(graph: dict[str, Any]) -> dict[str, int]:
    residue_ids = graph.get("residue_ids")
    chain_ids = graph.get("chain_ids")
    resseq = graph.get("resseq")
    mapping: dict[str, int] = {}
    count = int(graph["aa"].shape[0])
    for index in range(count):
        if residue_ids is not None and index < len(residue_ids):
            raw = str(residue_ids[index]).strip()
            if ":" in raw:
                mapping[raw.replace(" ", "")] = index
                continue
        if chain_ids is None or resseq is None:
            continue
        chain_id = str(chain_ids[index]).strip()
        resseq_value = int(resseq[index].item() if isinstance(resseq, torch.Tensor) else resseq[index])
        mapping[f"{chain_id}:{resseq_value}"] = index
    return mapping


def _check_mutation_aa_differs(item: dict[str, Any]) -> tuple[bool, str]:
    try:
        specs = parse_mutation_list(str(item.get("pdb_id", "")), str(item.get("mutation", "")))
    except Exception as exc:
        return False, f"could not parse mutation: {type(exc).__name__}: {exc}"

    wt_graph = item["wt_graph"]
    mt_graph = item["mt_graph"]
    wt_by_key = _residue_index_by_key(wt_graph)
    mt_by_key = _residue_index_by_key(mt_graph)
    for spec in specs:
        key = f"{spec.chain_id}:{spec.resseq}"
        if key not in wt_by_key or key not in mt_by_key:
            return False, f"mutation residue {key} missing from WT or MT graph"
        wt_aa = int(wt_graph["aa"][wt_by_key[key]].item())
        mt_aa = int(mt_graph["aa"][mt_by_key[key]].item())
        expected_wt = AA_TO_INDEX.get(spec.wt)
        expected_mt = AA_TO_INDEX.get(spec.mt)
        if expected_wt is not None and wt_aa != expected_wt:
            return False, f"WT mutation residue {key} aa={wt_aa} expected={expected_wt}"
        if expected_mt is not None and mt_aa != expected_mt:
            return False, f"MT mutation residue {key} aa={mt_aa} expected={expected_mt}"
        if wt_aa == mt_aa:
            return False, f"WT and MT mutation residue {key} have identical aa index {wt_aa}"
    return True, ""


def _coordinates_are_identical(item: dict[str, Any]) -> bool:
    wt_pos = item["wt_graph"].get("pos_atoms")
    mt_pos = item["mt_graph"].get("pos_atoms")
    if not isinstance(wt_pos, torch.Tensor) or not isinstance(mt_pos, torch.Tensor):
        return False
    if wt_pos.shape != mt_pos.shape:
        return False
    finite = torch.isfinite(wt_pos) & torch.isfinite(mt_pos)
    if not bool(finite.any().item()):
        return False
    return bool(torch.allclose(wt_pos[finite], mt_pos[finite]))


def _check_dataset_masks(dataset: Any, index: int) -> tuple[bool, str]:
    try:
        item = dataset[index]
    except Exception as exc:
        return False, f"paired dataset failed to build mutation masks: {type(exc).__name__}: {exc}"
    for state_key in ("wt_graph", "mt_graph"):
        mask = item[state_key].get("mutation_mask")
        if not isinstance(mask, torch.Tensor) or not bool(mask.to(dtype=torch.bool).any().item()):
            return False, f"{state_key} mutation_mask is missing or empty"
    return True, ""


_SHARED_GEOMETRY_TOKEN = "shared_wt_geometry_residue_identity"


def _uses_declared_shared_geometry(item: dict[str, Any]) -> bool:
    return _SHARED_GEOMETRY_TOKEN in str(item.get("construction_method", ""))


def _shared_geometry_contract(item: dict[str, Any]) -> tuple[bool, str]:
    if not _coordinates_are_identical(item):
        return False, "declared shared-geometry sample has different WT/MT coordinates"
    for state_key in ("wt_graph", "mt_graph"):
        metadata = item[state_key].get("metadata")
        if not isinstance(metadata, dict):
            return False, f"{state_key} is missing construction metadata"
        if _SHARED_GEOMETRY_TOKEN not in str(
            metadata.get("construction_method", "")
        ):
            return False, f"{state_key} does not declare shared-geometry construction"
    return True, ""


def audit_paired_ddg_cache(
    dataset: Any,
    *,
    max_examples: int = 8,
    strict: bool = True,
) -> dict[str, Any]:
    """Audit modeled pairs and explicitly declared shared-geometry pairs."""

    paths = [Path(path) for path in getattr(dataset, "paths", [])]
    report = _empty_report(len(paths))
    report["checks"]["declared_shared_geometry_contract"] = {"failures": 0}
    report["shared_geometry_exemptions"] = 0
    failed_samples: set[str] = set()

    for index, path in enumerate(paths):
        item = torch.load(path, map_location="cpu")
        if _uses_declared_shared_geometry(item):
            report["shared_geometry_exemptions"] += 1
            contract_ok, contract_reason = _shared_geometry_contract(item)
            if not contract_ok:
                _add_failure(
                    report,
                    failed_samples,
                    check="declared_shared_geometry_contract",
                    item=item,
                    path=path,
                    reason=contract_reason,
                    max_examples=max_examples,
                )
        else:
            mt_pdb = str(item.get("mt_pdb", ""))
            if not mt_pdb or not Path(mt_pdb).exists():
                _add_failure(
                    report,
                    failed_samples,
                    check="mt_pdb_exists",
                    item=item,
                    path=path,
                    reason=f"missing mt_pdb: {mt_pdb}",
                    max_examples=max_examples,
                )
            if _coordinates_are_identical(item):
                _add_failure(
                    report,
                    failed_samples,
                    check="wt_mt_coordinates_differ",
                    item=item,
                    path=path,
                    reason=(
                        "WT and MT pos_atoms tensors are identical without "
                        "a shared-geometry declaration"
                    ),
                    max_examples=max_examples,
                )

        mutation_ok, mutation_reason = _check_mutation_aa_differs(item)
        if not mutation_ok:
            _add_failure(
                report,
                failed_samples,
                check="mutation_aa_differs",
                item=item,
                path=path,
                reason=mutation_reason,
                max_examples=max_examples,
            )

        masks_ok, masks_reason = _check_dataset_masks(dataset, index)
        if not masks_ok:
            _add_failure(
                report,
                failed_samples,
                check="mutation_masks_present",
                item=item,
                path=path,
                reason=masks_reason,
                max_examples=max_examples,
            )

    report["num_failures"] = len(failed_samples)
    if strict and report["num_failures"]:
        raise ValueError(
            f"paired cache audit failed for {report['num_failures']} sample(s)"
        )
    return report
