from __future__ import annotations
import logging

import numpy as np
import pandas as pd

# =============================================================================
# VolatilityTiming — allocation dynamique depuis le spread s_t
# =============================================================================

class VolatilityTiming:
    """Convertit le spread s_t en signal d'allocation et l'applique aux positions.

    Intuition
    ---------
    Le spread s_t = σ_IV − σ̂_t mesure la "cherté" de la volatilité implicite
    par rapport à la volatilité réalisée estimée. On l'utilise pour scaler
    dynamiquement les poids de la stratégie de base :
        w_t = w_base × f(s_t)

    Modes de normalisation
    ----------------------
    "linear"    → signal = clip(s_t / rolling_std(s_t), ±max_leverage)
                  Proportionnel au spread normalisé par sa volatilité récente.

    "rank"      → signal = percentile roulant recentré dans [−max_leverage, +max_leverage]
                  Robuste aux outliers, agnostique à l'unité de mesure de s_t.

    "threshold" → signal = +1 si s_t > +seuil, −1 si s_t < −seuil, 0 sinon
                  Binaire, adapté à un sizing discret (full in / out).

    Paramètres
    ----------
    scaling      : str    Mode de normalisation (défaut "linear").
    lookback     : int    Fenêtre de normalisation en jours (défaut 63 ~ 3 mois).
    max_leverage : float  Multiplicateur maximum en valeur absolue (défaut 2.0).
    threshold    : float  Seuil en points de vol pour le mode "threshold" (défaut 0.02).
    """

    def __init__(
        self,
        scaling: str = "linear",
        lookback: int = 63,
        max_leverage: float = 2.0,
        threshold: float = 0.02,
    ) -> None:
        if scaling not in ("linear", "rank", "threshold"):
            raise ValueError("scaling doit être 'linear', 'rank' ou 'threshold'.")
        self.scaling = scaling
        self.lookback = lookback
        self.max_leverage = max_leverage
        self.threshold = threshold

    def compute_signal(self, spread: pd.Series) -> pd.Series:
        """Normalise s_t en signal d'allocation dans [−max_leverage, +max_leverage].

        Paramètres
        ----------
        spread : pd.Series  s_t = σ_IV − σ̂_t

        Retourne
        --------
        pd.Series  Signal de timing (même index que spread.dropna()).
        """
        spread = spread.dropna()

        if self.scaling == "linear":
            # Écart-type roulant de s_t sur la fenêtre lookback
            s_std = (
                spread.rolling(self.lookback, min_periods=5)
                .std()
                .replace(0.0, np.nan)
            )
            # Signal = s_t normalisé par son écart-type récent, clipé à ±max_leverage
            signal = (spread / s_std).clip(-self.max_leverage, self.max_leverage)

        elif self.scaling == "rank":
            # Rang percentile roulant de s_t (0 = min historique, 1 = max historique)
            rank = spread.rolling(self.lookback, min_periods=5).rank(pct=True)
            # Recentrage : rang 0.5 → 0 (neutre), 1.0 → +max_leverage, 0.0 → −max_leverage
            signal = ((rank - 0.5) * 2.0 * self.max_leverage).clip(
                -self.max_leverage, self.max_leverage
            )

        else:  # threshold
            # Signal binaire : ±1 au-delà des bandes ±threshold, 0 à l'intérieur
            signal = pd.Series(
                np.where(
                    spread >  self.threshold,  1.0,
                    np.where(spread < -self.threshold, -1.0, 0.0),
                ),
                index=spread.index,
            )

        signal.name = "timing_signal"
        return signal

    def apply_timing(
        self,
        df_positions: pd.DataFrame,
        signal: pd.Series,
        lag_bdays: int = 1,
    ) -> pd.DataFrame:
        """Multiplie les poids de la stratégie par le signal de timing.

        Le signal est décalé d'un jour ouvré par défaut : le poids appliqué à la
        date t est donc construit à partir de l'information disponible en t-1.
        Cela évite qu'une position prise à la date t utilise déjà r_t ou σ̂_t.

        Les dates sans signal disponible reçoivent un multiplicateur 1.0 (neutre),
        pour ne pas introduire de gaps dans la série de positions.

        Paramètres
        ----------
        df_positions : pd.DataFrame  Sortie de OptionTrade.generate_trades.
        signal       : pd.Series     Sortie de compute_signal (index = dates).
        lag_bdays    : int           Décalage d'exécution en jours ouvrés.

        Retourne
        --------
        pd.DataFrame  Identique à l'entrée avec la colonne 'weight' rescalée.
        """
        df = df_positions.copy()

        signal_exec = signal.copy()
        if lag_bdays > 0:
            signal_exec.index = signal_exec.index + pd.offsets.BDay(lag_bdays)

        # Préparer le signal pour le merge (reset_index() car index = dates)
        signal_df = (
            signal_exec.rename("timing_signal")
            .reset_index()
            .rename(columns={"index": "date"})
        )

        # Left join : toutes les lignes de positions sont conservées
        df = df.merge(signal_df, on="date", how="left")

        # Dates sans signal → multiplicateur neutre 1.0 (pas de levier forcé)
        df["timing_signal"] = df["timing_signal"].fillna(1.0)

        # Application du timing : w_t = w_base × f(s_t)
        df["weight"] = df["weight"] * df["timing_signal"]
        df = df.drop(columns=["timing_signal"])

        logging.info(
            "Volatility timing appliqué à %d lignes avec un lag de %d jour(s) ouvré(s).",
            len(df),
            lag_bdays,
        )
        return df
