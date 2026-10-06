"""Command line interface for building MeerKAT primary beam models."""

import argparse
import os
import sys

import numpy as np

from .parallelize import parmap
from .spatial import recon_par
from .spectral import dct_recon_all
from .util import jones_to_mueller_all, write_fits, write_fits_eight

package_directory = os.path.dirname(__file__)

# Stokes pairings, indexed by [row, column]: entry 'IQ' is the leakage from Q to I.
STOKES_MATRIX = [
    ["I", "IQ", "IU", "IV"],
    ["QI", "Q", "QU", "QV"],
    ["UI", "UQ", "U", "UV"],
    ["VI", "VQ", "VU", "V"],
]
# Every selectable two-letter Stokes pairing, plus 'M' for the full Mueller matrix.
STOKES_COMBINATIONS = frozenset(s for line in STOKES_MATRIX for s in line)


def zernike_parameters(filename, npix=256, diameter=10, thr=20):
    C = np.load(filename, encoding="latin1", allow_pickle=True).item()
    Cr = dct_recon_all(C)
    diameter_orig = 10.0  # original coeffs were calculated from 10 deg beams
    Npix = int(diameter_orig / (diameter / npix))
    params = [[Cr[:, i, :, :, :], C["zi"], Npix, thr] for i in range(Cr.shape[1])]
    return np.array(params, dtype=object), C["nu"]


def save_fits(data, nu, args, filename, centre=None):
    # Save as fits files
    if args.output_eight and args.Stokes is None:
        write_fits_eight(data, nu, args.diameter, filename, centre=centre)
        print(f"Saved as 8 files with prefix {filename}")
    elif args.output_eight and args.Stokes is not None:
        print("8 output files can be created for Jones formalism only, not for Mueller")
        print("!WARNING: No output file created")
    else:
        write_fits(data.real, nu, args.diameter, filename + "_re.fits", centre=centre)
        print(f"Saved Real part as {filename}_re.fits")
        if args.Stokes is None:
            write_fits(
                data.imag, nu, args.diameter, filename + "_im.fits", centre=centre
            )
            print(f"Saved Imaginary part as {filename}_im.fits")


def build_parser():
    parser = argparse.ArgumentParser(description="Create primary beam model of MeerKAT")
    parser.add_argument(
        "-p", "--pixels", help="Number of pixels on one side", type=int, required=False
    )
    parser.add_argument(
        "-d",
        "--diameter",
        help="Diameter of the required beam",
        type=float,
        required=False,
    )
    parser.add_argument(
        "-r", "--scale", help="Pixel scale in degrees", type=float, required=False
    )
    parser.add_argument(
        "-f",
        "--freq",
        help="A single freq, or the start, end freqs, and channel width in MHz",
        nargs="+",
        type=float,
        required=True,
    )
    parser.add_argument(
        "-c",
        "--coeff",
        help="Which coefficients to use: mh for MeerKAT holography, me for MeerKAT EM "
        "simulation and vh for VLA holography?",
        type=str,
        default="mh",
        choices=("mh", "me", "vh"),
    )
    parser.add_argument("-cf", "--coefficients-file", help="Coefficients file")
    parser.add_argument(
        "-P",
        "--prefix",
        help="Prefix of output beam beam file(s)",
        type=str,
        required=False,
    )
    parser.add_argument(
        "-o8",
        "--output-eight",
        help="Output complex volatge beams (8 files)",
        action="store_true",
    )
    parser.add_argument(
        "-T",
        "--thresh",
        help="How many Zernike coefficients to use. Must be <=20.",
        type=int,
        default=20,
        required=False,
    )
    parser.add_argument(
        "-S",
        "--Stokes",
        help="If provided output will be in Stokes, i. e. Mueller, formalism, instead of the "
        "default Jones formalism. Specify 'I', 'Q', 'U', 'V' for the Stokes beams, or 'M' to "
        "get the full Mueller matrix. If you give 'IQ' instead, the leakage from Q to I will "
        "be provided, and so on for any other combination of Stokes parameters.",
        type=str,
        required=False,
    )
    return parser


def main(argv):
    parser = build_parser()
    args = parser.parse_args(argv)

    # create the list of frequencies
    if len(args.freq) == 1:
        nu = float(args.freq[0])
    elif len(args.freq) == 2:
        nu = np.arange(args.freq[0], args.freq[1], 1)
    elif len(args.freq) == 3:
        nu = np.arange(args.freq[0], args.freq[1], args.freq[2])
    else:
        parser.error("Do `eidos -h` to see how to input the frequency parameters")

    # Zernike coefficient filename
    if args.coefficients_file:
        filename = args.coefficients_file
    elif args.coeff == "mh":
        filename = os.path.join(
            package_directory, "data", "meerkat_beam_coeffs_ah_zp_dct.npy"
        )
    elif args.coeff == "me":
        filename = os.path.join(
            package_directory, "data", "meerkat_beam_coeffs_em_zp_dct.npy"
        )
    else:
        raise NotImplementedError("JVLA option is coming soon")

    # pixel diameter and scale
    if not args.pixels:
        if args.diameter is None or args.scale is None:
            parser.error("Specify both diameter and pixel scale")
        args.pixels = int(args.diameter / args.scale)
    if not args.diameter:
        if args.pixels is None or args.scale is None:
            parser.error("Specify both number of pixels and pixel scale")
        args.diameter = args.pixels * args.scale

    # Create parameter list for Zernike reconstruction
    params, freqs = zernike_parameters(
        filename, args.pixels, args.diameter, args.thresh
    )

    # Create beam Jones matrix using Zernike polynomials within 10 degrees
    if isinstance(nu, (int, float)):
        ch = abs(freqs - nu).argmin()
        B = recon_par(params[ch, :])
    else:
        ch = [abs(freqs - i).argmin() for i in nu]
        B = np.array(parmap(recon_par, params[ch, :]))

    # Cut the beam to the specified diameter
    if len(B.shape) == 4:
        B = np.expand_dims(B, axis=0)
    # 0-based pixel of the beam centre: n / 2 on the reconstruction grid (spatial.Zernike.unit_disk)
    centre = B.shape[-1] / 2.0
    if args.diameter != 10:
        c, r = int(B.shape[-1] / 2), int(args.pixels / 2)
        centre -= c - r  # the crop window starts at pixel c - r
        if args.pixels % 2 == 0:
            B = B[..., c - r : c + r, c - r : c + r]
        else:
            B = B[..., c - r : c + r + 1, c - r : c + r + 1]
    print("Cube shape", B.shape)

    if args.prefix:
        filename = args.prefix
    else:
        if isinstance(nu, np.ndarray):
            chan = f"{len(nu)}channels"
        else:
            chan = f"{int(nu)}MHz"
        filename = f"primary_beam_{args.coeff}_{chan}_{args.diameter:g}deg"

    # Convert to Stokes/Mueller formalism
    if args.Stokes:
        m = args.Stokes
        if m != "M" and m not in STOKES_COMBINATIONS:
            parser.error(
                "--Stokes must be 'M' or one of "
                + ", ".join(sorted(STOKES_COMBINATIONS))
            )
        data_M = jones_to_mueller_all(B)
        if m == "M":
            data = data_M
        else:
            ind = np.where(np.array(STOKES_MATRIX) == m)
            data = np.zeros((B.shape[0], 1, 1, B.shape[3], B.shape[4]), dtype=complex)
            data[:, 0, 0, :, :] = data_M[:, ind[0][0], ind[1][0], ...]
        filename = filename + "_" + m
        save_fits(data, nu, args, filename, centre=centre)
    else:
        save_fits(B, nu, args, filename, centre=centre)


def cli():
    """Entry point for the ``eidos`` console script."""
    main(sys.argv[1:])


if __name__ == "__main__":
    cli()
