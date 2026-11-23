"""
soundml/backend/data/dataset.py
─────────────────────────────────
Generates and manages the Spotify-style audio features dataset.

Each track has 6 Spotify Audio Analysis features:
  • danceability  [0.0 – 1.0]  Suitability for dancing
  • energy        [0.0 – 1.0]  Intensity/activity measure
  • valence       [0.0 – 1.0]  Musical positiveness
  • tempo         [60  – 200]  BPM estimate
  • acousticness  [0.0 – 1.0]  Acoustic vs electronic
  • speechiness   [0.0 – 1.0]  Spoken word presence

Genres: Pop, Hip-Hop, Electronic, Rock, R&B, Country
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Genre-specific feature distributions (mean, std) ──────────────────────
# These distributions are carefully calibrated to match
# real Spotify data patterns per genre.

GENRE_DISTRIBUTIONS = {
    "Pop": {
        "danceability": (0.63, 0.12),
        "energy":       (0.66, 0.13),
        "valence":      (0.58, 0.18),
        "tempo":        (118,  22),
        "acousticness": (0.18, 0.15),
        "speechiness":  (0.05, 0.02),
    },
    "Hip-Hop": {
        "danceability": (0.74, 0.10),
        "energy":       (0.63, 0.14),
        "valence":      (0.49, 0.17),
        "tempo":        (120,  25),
        "acousticness": (0.12, 0.10),
        "speechiness":  (0.17, 0.07),   # ← Key separator
    },
    "Electronic": {
        "danceability": (0.60, 0.11),
        "energy":       (0.84, 0.10),   # ← Key separator
        "valence":      (0.55, 0.20),
        "tempo":        (128,  10),     # ← Very consistent
        "acousticness": (0.02, 0.02),   # ← Key separator
        "speechiness":  (0.03, 0.01),
    },
    "Rock": {
        "danceability": (0.46, 0.12),   # ← Key separator
        "energy":       (0.78, 0.15),
        "valence":      (0.42, 0.18),
        "tempo":        (128,  28),
        "acousticness": (0.08, 0.10),
        "speechiness":  (0.04, 0.02),
    },
    "R&B": {
        "danceability": (0.68, 0.12),
        "energy":       (0.52, 0.16),   # ← Key separator (lower)
        "valence":      (0.51, 0.17),
        "tempo":        (103,  22),
        "acousticness": (0.28, 0.20),
        "speechiness":  (0.05, 0.02),
    },
    "Country": {
        "danceability": (0.52, 0.10),
        "energy":       (0.57, 0.15),
        "valence":      (0.62, 0.18),
        "tempo":        (98,   20),
        "acousticness": (0.40, 0.22),   # ← Key separator
        "speechiness":  (0.04, 0.01),
    },
}

GENRES = list(GENRE_DISTRIBUTIONS.keys())
FEATURES = ["danceability", "energy", "valence", "tempo", "acousticness", "speechiness"]

# ── Real track anchors (ground truth) ─────────────────────────────────────
REAL_TRACKS = [
    # Pop
    ("As It Was", "Harry Styles", "Pop", 0.52, 0.73, 0.56, 174, 0.05, 0.03),
    ("Flowers", "Miley Cyrus", "Pop", 0.67, 0.68, 0.76, 117, 0.18, 0.04),
    ("Anti-Hero", "Taylor Swift", "Pop", 0.64, 0.64, 0.42, 97, 0.07, 0.05),
    ("Levitating", "Dua Lipa", "Pop", 0.70, 0.82, 0.92, 103, 0.02, 0.04),
    ("Blinding Lights", "The Weeknd", "Pop", 0.51, 0.80, 0.33, 171, 0.00, 0.06),
    ("Good 4 U", "Olivia Rodrigo", "Pop", 0.56, 0.89, 0.67, 166, 0.01, 0.06),
    ("Shape of You", "Ed Sheeran", "Pop", 0.83, 0.65, 0.93, 96, 0.08, 0.08),
    ("Stay", "Justin Bieber", "Pop", 0.82, 0.65, 0.65, 170, 0.16, 0.05),
    # Hip-Hop
    ("Rich Flex", "Drake & 21 Savage", "Hip-Hop", 0.72, 0.63, 0.37, 109, 0.07, 0.22),
    ("Industry Baby", "Lil Nas X", "Hip-Hop", 0.74, 0.75, 0.64, 149, 0.01, 0.12),
    ("God's Plan", "Drake", "Hip-Hop", 0.75, 0.44, 0.43, 77, 0.21, 0.11),
    ("Sicko Mode", "Travis Scott", "Hip-Hop", 0.75, 0.77, 0.44, 155, 0.04, 0.22),
    ("HUMBLE.", "Kendrick Lamar", "Hip-Hop", 0.91, 0.62, 0.42, 150, 0.05, 0.22),
    ("DNA.", "Kendrick Lamar", "Hip-Hop", 0.79, 0.79, 0.47, 141, 0.10, 0.26),
    ("Hotline Bling", "Drake", "Hip-Hop", 0.69, 0.36, 0.66, 135, 0.09, 0.06),
    # Electronic
    ("Levels", "Avicii", "Electronic", 0.67, 0.95, 0.89, 126, 0.01, 0.03),
    ("Animals", "Martin Garrix", "Electronic", 0.57, 0.93, 0.40, 128, 0.00, 0.04),
    ("Clarity", "Zedd", "Electronic", 0.55, 0.88, 0.45, 128, 0.00, 0.03),
    ("Faded", "Alan Walker", "Electronic", 0.59, 0.80, 0.22, 90, 0.05, 0.02),
    ("Titanium", "David Guetta", "Electronic", 0.59, 0.86, 0.46, 126, 0.01, 0.03),
    ("Midnight City", "M83", "Electronic", 0.55, 0.67, 0.37, 106, 0.02, 0.03),
    # Rock
    ("Mr. Brightside", "The Killers", "Rock", 0.34, 0.92, 0.25, 148, 0.00, 0.05),
    ("Bohemian Rhapsody", "Queen", "Rock", 0.38, 0.40, 0.22, 72, 0.11, 0.05),
    ("Smells Like Teen Spirit", "Nirvana", "Rock", 0.50, 0.92, 0.44, 117, 0.00, 0.05),
    ("Wonderwall", "Oasis", "Rock", 0.35, 0.50, 0.43, 87, 0.25, 0.03),
    ("Highway to Hell", "AC/DC", "Rock", 0.45, 0.95, 0.60, 116, 0.01, 0.04),
    ("Creep", "Radiohead", "Rock", 0.26, 0.52, 0.09, 92, 0.17, 0.03),
    # R&B
    ("Leave The Door Open", "Silk Sonic", "R&B", 0.67, 0.35, 0.49, 96, 0.27, 0.04),
    ("All of Me", "John Legend", "R&B", 0.39, 0.18, 0.36, 120, 0.86, 0.03),
    ("Redbone", "Childish Gambino", "R&B", 0.62, 0.47, 0.52, 96, 0.03, 0.03),
    ("Location", "Khalid", "R&B", 0.70, 0.42, 0.50, 93, 0.44, 0.04),
    ("Drunk in Love", "Beyoncé", "R&B", 0.72, 0.61, 0.37, 131, 0.14, 0.06),
    # Country
    ("Jolene", "Dolly Parton", "Country", 0.47, 0.44, 0.48, 112, 0.62, 0.03),
    ("Body Like a Back Road", "Sam Hunt", "Country", 0.63, 0.65, 0.80, 84, 0.23, 0.04),
    ("Friends in Low Places", "Garth Brooks", "Country", 0.52, 0.61, 0.77, 98, 0.33, 0.04),
    ("Take Me Home, Country Roads", "John Denver", "Country", 0.40, 0.44, 0.72, 83, 0.75, 0.03),
    ("Tennessee Whiskey", "Chris Stapleton", "Country", 0.42, 0.44, 0.55, 80, 0.39, 0.03),
    ("Old Town Road", "Lil Nas X", "Country", 0.76, 0.67, 0.68, 136, 0.15, 0.09),
]


def generate_synthetic_tracks(n_per_genre: int = 50, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic tracks by sampling from genre-specific
    Gaussian distributions, then clipping to valid ranges.

    Args:
        n_per_genre: Number of synthetic tracks per genre
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: genre + FEATURES
    """
    rng = np.random.default_rng(seed)
    rows = []

    for genre, dist in GENRE_DISTRIBUTIONS.items():
        for i in range(n_per_genre):
            row = {"title": f"Synthetic_{genre}_{i+1}", "artist": "Synthetic", "genre": genre}
            for feature, (mean, std) in dist.items():
                val = rng.normal(mean, std)
                if feature == "tempo":
                    val = float(np.clip(val, 60, 200))
                else:
                    val = float(np.clip(val, 0.0, 1.0))
                row[feature] = round(val, 4)
            rows.append(row)

    return pd.DataFrame(rows)


def load_real_tracks() -> pd.DataFrame:
    """Load the curated real-track anchor points."""
    columns = ["title", "artist", "genre"] + FEATURES
    return pd.DataFrame(REAL_TRACKS, columns=columns)


def build_full_dataset(
    n_synthetic_per_genre: int = 50,
    seed: int = 42,
    save_path: Path | None = None,
) -> pd.DataFrame:
    """
    Build the complete dataset: real tracks + synthetic tracks.

    Args:
        n_synthetic_per_genre: Synthetic samples per genre
        seed: Reproducibility seed
        save_path: If provided, save CSV to this path

    Returns:
        Full DataFrame ready for ML training
    """
    real = load_real_tracks()
    synthetic = generate_synthetic_tracks(n_per_genre=n_synthetic_per_genre, seed=seed)

    df = pd.concat([real, synthetic], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # Shuffle

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"✅ Dataset saved: {save_path} ({len(df)} rows)")

    return df


def get_dataset_stats(df: pd.DataFrame) -> dict:
    """Compute descriptive statistics for the dataset."""
    stats = {
        "total_tracks": len(df),
        "genres": GENRES,
        "features": FEATURES,
        "genre_counts": df["genre"].value_counts().to_dict(),
        "feature_stats": {},
    }
    for f in FEATURES:
        stats["feature_stats"][f] = {
            "mean": round(float(df[f].mean()), 4),
            "std":  round(float(df[f].std()), 4),
            "min":  round(float(df[f].min()), 4),
            "max":  round(float(df[f].max()), 4),
        }
    return stats


if __name__ == "__main__":
    from backend.utils.config import settings
    df = build_full_dataset(save_path=settings.DATASET_PATH)
    print(f"Total samples: {len(df)}")
    print(df["genre"].value_counts())
    print(df.describe())
