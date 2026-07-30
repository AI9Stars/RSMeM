#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Part of the RSMeM evaluation suite.
"""NEI (Network Efficiency Index): Acc * (1 - C_refl / C_total).

Captures how much accuracy is retained per unit of reflection-token overhead.
"""

import argparse


def calculate_nei(acc: float, c_refl: float, c_total: float) -> float:
    """Return NEI = acc * (1 - c_refl / c_total), rounded to 4 decimals.

    acc:     accuracy value (unit is preserved in the output)
    c_refl:  reflection token usage/count
    c_total: total token usage/count
    """
    if c_total <= 0:
        raise ValueError("c_total must be positive")
    rho = c_refl / c_total
    return round(acc * (1 - rho), 4)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute NEI = Acc * (1 - C_refl / C_total).",
    )
    parser.add_argument("--acc", type=float, required=True, help="Accuracy value")
    parser.add_argument("--c-refl", type=float, required=True, help="Reflection token usage/count")
    parser.add_argument("--c-total", type=float, required=True, help="Total token usage/count")
    args = parser.parse_args()

    print(f"NEI: {calculate_nei(args.acc, args.c_refl, args.c_total)}")


if __name__ == "__main__":
    main()
