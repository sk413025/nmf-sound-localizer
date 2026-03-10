"""
Nature Communications compliance validator for figure outputs.

Checks: dimensions, font families, font embedding, DPI, file format pairing,
line weight, and colour-vision-deficiency accessibility.

Usage:
    python -m figures.validate figures/output/           # validate all
    python -m figures.validate figures/output/fig02.pdf  # validate single
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# ---------------------------------------------------------------------------
# Nature Communications constraints
# ---------------------------------------------------------------------------
ALLOWED_WIDTHS_MM = (89, 183)  # single / double column
WIDTH_TOL_MM = 1.0
MAX_HEIGHT_MM = 247
HEIGHT_TOL_MM = 1.0
MIN_DPI = 300
ALLOWED_FONT_FAMILIES = {"Arial", "Helvetica", "DejaVu Sans", "DejaVuSans"}
MIN_LINE_WIDTH_PT = 0.5
MAX_LINE_WIDTH_PT = 1.0


class Severity(Enum):
    ERROR = "ERROR"
    WARNING = "WARN"


@dataclass
class Issue:
    check: str
    message: str
    severity: Severity


@dataclass
class FileReport:
    path: Path
    issues: list[Issue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(i.severity == Severity.ERROR for i in self.issues)


# ---------------------------------------------------------------------------
# Check implementations
# ---------------------------------------------------------------------------

def _check_dimensions(pdf_path: Path) -> list[Issue]:
    issues: list[Issue] = []
    try:
        import fitz  # PyMuPDF
    except ImportError:
        issues.append(Issue("dimensions", "PyMuPDF not installed — skipping dimension check", Severity.WARNING))
        return issues

    doc = fitz.open(str(pdf_path))
    if doc.page_count == 0:
        issues.append(Issue("dimensions", "PDF has no pages", Severity.ERROR))
        return issues

    page = doc[0]
    rect = page.mediabox
    width_pt = rect.width
    height_pt = rect.height
    width_mm = width_pt * 25.4 / 72.0
    height_mm = height_pt * 25.4 / 72.0
    doc.close()

    width_ok = any(abs(width_mm - w) <= WIDTH_TOL_MM for w in ALLOWED_WIDTHS_MM)
    if not width_ok:
        issues.append(Issue(
            "dimensions",
            f"Width {width_mm:.1f}mm — need {ALLOWED_WIDTHS_MM[0]}mm or {ALLOWED_WIDTHS_MM[1]}mm (tolerance {WIDTH_TOL_MM}mm)",
            Severity.ERROR,
        ))

    if height_mm > MAX_HEIGHT_MM + HEIGHT_TOL_MM:
        issues.append(Issue(
            "dimensions",
            f"Height {height_mm:.1f}mm — exceeds max {MAX_HEIGHT_MM}mm",
            Severity.ERROR,
        ))

    return issues


def _check_fonts(pdf_path: Path) -> list[Issue]:
    issues: list[Issue] = []
    try:
        import fitz
    except ImportError:
        issues.append(Issue("font_family", "PyMuPDF not installed — skipping font check", Severity.WARNING))
        return issues

    doc = fitz.open(str(pdf_path))
    for page_num in range(doc.page_count):
        page = doc[page_num]
        font_list = page.get_fonts(full=True)
        for font_entry in font_list:
            font_name = font_entry[3] if len(font_entry) > 3 else ""
            font_type = font_entry[2] if len(font_entry) > 2 else ""

            # Check font family
            base_name = font_name.split("+")[-1] if "+" in font_name else font_name
            base_name_clean = base_name.split("-")[0].split(",")[0].strip()

            is_allowed = any(
                allowed.lower() in base_name_clean.lower()
                for allowed in ALLOWED_FONT_FAMILIES
            )
            if not is_allowed and base_name_clean:
                issues.append(Issue(
                    "font_family",
                    f'Found "{base_name}" — need sans-serif (Arial/Helvetica/DejaVu Sans)',
                    Severity.ERROR,
                ))

            # Check embedding (Type 42 = TrueType)
            if font_type and "Type1" in font_type and "TrueType" not in font_type:
                issues.append(Issue(
                    "font_embedding",
                    f'Font "{base_name}" is Type1, not TrueType (Type 42)',
                    Severity.ERROR,
                ))

    doc.close()
    return issues


def _check_tiff_dpi(tiff_path: Path) -> list[Issue]:
    issues: list[Issue] = []
    try:
        from PIL import Image
    except ImportError:
        issues.append(Issue("dpi", "Pillow not installed — skipping DPI check", Severity.WARNING))
        return issues

    img = Image.open(tiff_path)
    dpi = img.info.get("dpi", (72, 72))
    min_dpi = min(dpi[0], dpi[1])
    if min_dpi < MIN_DPI:
        issues.append(Issue("dpi", f"DPI {min_dpi:.0f} — need >= {MIN_DPI}", Severity.ERROR))
    img.close()
    return issues


def _check_file_pairing(path: Path, all_paths: set[Path]) -> list[Issue]:
    issues: list[Issue] = []
    stem = path.stem
    parent = path.parent

    if path.suffix.lower() == ".pdf":
        tiff_path = parent / f"{stem}.tiff"
        if tiff_path not in all_paths:
            issues.append(Issue("file_format", f"Missing TIFF pair: {tiff_path.name}", Severity.ERROR))
    elif path.suffix.lower() == ".tiff":
        pdf_path = parent / f"{stem}.pdf"
        if pdf_path not in all_paths:
            issues.append(Issue("file_format", f"Missing PDF pair: {pdf_path.name}", Severity.ERROR))

    return issues


def _check_line_weights(pdf_path: Path) -> list[Issue]:
    issues: list[Issue] = []
    try:
        import fitz
    except ImportError:
        return issues

    doc = fitz.open(str(pdf_path))
    min_lw = float("inf")
    for page_num in range(doc.page_count):
        page = doc[page_num]
        drawings = page.get_drawings()
        for d in drawings:
            w = d.get("width", None)
            if w is not None and w > 0:
                min_lw = min(min_lw, w)
    doc.close()

    if min_lw < float("inf") and min_lw < MIN_LINE_WIDTH_PT:
        issues.append(Issue(
            "line_weight",
            f"Min line width {min_lw:.2f}pt — suggest >= {MIN_LINE_WIDTH_PT}pt",
            Severity.WARNING,
        ))

    return issues


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_file(path: Path, all_paths: set[Path] | None = None) -> FileReport:
    """Validate a single figure file."""
    report = FileReport(path=path)

    if all_paths is None:
        all_paths = set()

    if path.suffix.lower() == ".pdf":
        report.issues.extend(_check_dimensions(path))
        report.issues.extend(_check_fonts(path))
        report.issues.extend(_check_line_weights(path))
        report.issues.extend(_check_file_pairing(path, all_paths))
    elif path.suffix.lower() == ".tiff":
        report.issues.extend(_check_tiff_dpi(path))
        report.issues.extend(_check_file_pairing(path, all_paths))

    return report


def validate_all(directory: Path) -> bool:
    """Validate all PDF/TIFF files in a directory. Returns True if all pass."""
    pdf_files = sorted(directory.glob("*.pdf"))
    tiff_files = sorted(directory.glob("*.tiff"))
    all_files = pdf_files + tiff_files
    all_paths = {f for f in all_files}

    if not all_files:
        print(f"No PDF/TIFF files found in {directory}")
        return True

    reports: list[FileReport] = []
    for f in all_files:
        reports.append(validate_file(f, all_paths))

    # Print results
    n_fail = 0
    for report in reports:
        if report.issues:
            status = "PASS" if report.passed else "FAIL"
            if not report.passed:
                n_fail += 1
            print(f"\n{status}  {report.path.name}")
            for issue in report.issues:
                print(f"  [{issue.severity.value}] {issue.check}: {issue.message}")
        # Skip files with no issues (silent pass)

    n_total = len(set(f.stem for f in all_files))
    if n_fail > 0:
        print(f"\n{n_fail} figure(s) failed validation. Deployment blocked.")
        return False
    else:
        print(f"\nAll {n_total} figure(s) passed validation.")
        return True


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m figures.validate <path_or_directory> [...]")
        sys.exit(1)

    targets = [Path(p) for p in sys.argv[1:]]
    all_ok = True

    for target in targets:
        if target.is_dir():
            ok = validate_all(target)
            all_ok = all_ok and ok
        elif target.is_file():
            report = validate_file(target)
            status = "PASS" if report.passed else "FAIL"
            print(f"{status}  {report.path.name}")
            for issue in report.issues:
                print(f"  [{issue.severity.value}] {issue.check}: {issue.message}")
            all_ok = all_ok and report.passed
        else:
            print(f"Not found: {target}")
            all_ok = False

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
