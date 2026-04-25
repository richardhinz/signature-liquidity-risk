Python implementation of the signature-based VaR framework. It simulates price paths, computes iterated integrals (Lévy area), and applies Elastic Net to map path geometry to liquidity loss. The script empirically proves that Level-2 signatures capture the "speed and shape" of crashes, correcting standard Markovian underestimation.

Project Overview
This repository contains the empirical validation and algorithmic core for the research paper: "Non-Markovian Liquidity Risk via Rough Path Signatures" (Hinz, 2026).

The project challenges the Markovian assumption central to Basel III/IV regulatory capital (the square-root-of-time scaling rule). We demonstrate that by using Rough Path Signatures, banks can encode the "geometry" of a market crash—specifically its acceleration and path-dependency—to predict liquidity shortfalls that traditional models miss.

Key Mathematical Contributions
1. The Theoretical Unification (Bridging Markov and Rough Paths)
We establish that standard, path-blind VaR is a Level-1 degenerate case of a signature-based risk measure. This implementation proves that moving to Level-2 signatures (encoding quadratic variation and path-shape) provides the "High-Definition" upgrade required for non-linear market regimes.

2. The Geometric Crisis Signal (Lévy-Area Correction)
The algorithm identifies the Lévy Area as a model-free proxy for liquidity risk. Unlike the standard square-root-of-time rule, which assumes independent increments, our signature-based estimator captures how the "winding" and "velocity" of a price path amplify market impact and capital depletion.

Technical Implementation
Lead-Lag Transformation: Augments 1D price series into a multi-dimensional path to ensure non-vanishing signature components.

Iterated Integrals: Efficient computation of the truncated signature Sig2(X) to extract geometric features.

Elastic Net Estimator: A robust, regularized mapping from the signature feature space to observed liquidity-adjusted losses.

VaR Gap Analysis: Statistical comparison between the "True" path-dependent risk and the underestimated "Markovian" risk.

Results
As detailed in Section 6 of the paper, the signature-based correction reduces VaR underestimation during "Fast Crash" scenarios by identifying the hidden "Risk-Path" signal that exists beyond simple volatility.

References
Hinz, R. E. (2026). Non-Markovian Liquidity Risk via Rough Path Signatures: A Model-Free Framework for Bank Capital.
