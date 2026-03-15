from typing import Dict, List, Optional

LABELS: List[str] = [
    "Person",
    "Location",
    "Organization",
    "Date",
    "Time",
    "Animal",
    "Quantity",
    "Event",
    "LocationCountry",
    "LocationCity",
    "Shop",
    "CultureSite",
    "Building",
    "Duraion",
    "TimeDuration",
    "Sports",
    "Food",
    "Currency",
    "Law",
    "QuantityAge",
    "QunatityTemperature",
    "QuantityPrice",
    "EventSports",
    "EventFestival",
    "TermDisease",
    "TermSports",
]

# Dataset-side aliases for labels that differ from the target label list.
DATASET_LABEL_ALIASES: Dict[str, str] = {
    "CulturalAsset": "CultureSite",
    "DateDuration": "Duraion",
    "QuantityTemperature": "QunatityTemperature",
    "QuantityMoney": "QuantityPrice",
}


def normalize_dataset_label(raw_label: str) -> Optional[str]:
    canonical = DATASET_LABEL_ALIASES.get(raw_label, raw_label)
    if canonical in LABELS:
        return canonical
    return None


def build_bio_labels(base_labels: Optional[List[str]] = None) -> List[str]:
    labels = base_labels or LABELS
    return ["O"] + [f"B-{label}" for label in labels] + [f"I-{label}" for label in labels]
