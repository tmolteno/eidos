"""Some utility functions to be used by other scripts."""

import time

import numpy as np
from astropy.io import fits


def normalise(d):
    xmid = round(d.shape[2] / 2.0) - 1
    ymid = round(d.shape[3] / 2.0) - 1
    m = d[:, :, xmid, ymid]
    m = np.linalg.inv(m)
    for i in range(d.shape[2]):
        for j in range(d.shape[3]):
            d[:, :, i, j] = np.dot(m, d[:, :, i, j])
    return d


def normalise_multifreq(d):
    data = np.zeros(d.shape, dtype=complex)
    for i in range(d.shape[2]):
        data[:, :, i, :, :] = normalise(d[:, :, i, :, :])
    return data


def jones_to_mueller(a, b=None):
    """
    Convert 2x2xNxN Jones matrix into 4x4xNxN Mueller matrix.
    If f1=f2, compute the autocorrelation version for single dishes
    """
    if b is None:
        b = a
    M = np.zeros((4, 4, a.shape[2], a.shape[3]), dtype=complex)
    S = 0.5 * np.array([[1, 1, 0, 0], [0, 0, 1, 1j], [0, 0, 1, -1j], [1, -1, 0, 0]])
    S_inv = np.linalg.inv(S)
    for i in range(a.shape[2]):
        for j in range(a.shape[3]):
            ab = np.kron(a[:, :, i, j], b[:, :, i, j].conj())
            M[:, :, i, j] = np.dot(np.dot(S_inv, ab), S)
    return M


def jones_to_mueller_all(d):
    M = np.zeros((d.shape[0], 4, 4, d.shape[3], d.shape[4]), dtype=complex)
    for f in range(d.shape[0]):
        M[f, :, :, :, :] = jones_to_mueller(d[f, :, :, :, :])
    return M


def write_fits(beam, freqs, diameter, filename):
    # Create header
    hdr = fits.Header()
    fMHz = np.array(freqs) * 1e6
    if isinstance(fMHz, (int, float)):
        fMHz = [fMHz]
    try:
        df = fMHz[1] - fMHz[0]
    except (IndexError, TypeError):
        df = 1e6
    diam = float(diameter)
    if beam.shape[0] == 2:
        xy = ["H", "V"]
    elif beam.shape[0] == 4:
        xy = ["Mx", "My"]
    else:
        xy = ["", ""]
    ctypes = ["FREQ", xy[0], xy[1], "px", "py"][::-1]
    crvals = [fMHz[0], 0, 0, 0.0, 0.0][::-1]
    cdelts = [df, 1, 1, diam / beam.shape[-2], diam / beam.shape[-1]][::-1]
    cunits = ["Hz", "", "", "deg", "deg"][::-1]
    nx, ny = beam.shape[-2], beam.shape[-1]
    if nx % 2 == 0:
        crpixx, crpixy = nx / 2, ny / 2
    elif nx % 2 == 1:
        crpixx, crpixy = int(nx / 2), int(ny / 2)
    crpixs = [1, 1, 1, crpixy, crpixx][::-1]
    for i in range(len(beam.shape)):
        ii = str(i + 1)
        hdr["CTYPE" + ii] = ctypes[i]
        hdr["CRPIX" + ii] = crpixs[i]
        hdr["CRVAL" + ii] = crvals[i]
        hdr["CDELT" + ii] = cdelts[i]
        hdr["CUNIT" + ii] = cunits[i]
    hdr["TELESCOP"] = "MeerKAT"
    hdr["DATE"] = time.ctime()

    # Write real and imag parts of data
    hdu = fits.PrimaryHDU(beam, header=hdr)
    hdu.writeto(filename, overwrite=True)


def write_fits_cube(beam, freqs, diameter, filename):
    # Create header
    hdr = fits.Header()
    fMHz = np.array(freqs) * 1e6
    if isinstance(fMHz, (int, float)):
        fMHz = [fMHz]
    try:
        df = fMHz[1] - fMHz[0]
    except (IndexError, TypeError):
        df = 1e6
    diam = float(diameter)
    ctypes = ["px", "py", "FREQ"]
    crvals = [0.0, 0.0, fMHz[0]]
    cdelts = [diam / beam.shape[-2], diam / beam.shape[-1], df]
    cunits = ["deg", "deg", "Hz"]
    nx, ny = beam.shape[-2], beam.shape[-1]
    if nx % 2 == 0:
        crpixx, crpixy = nx / 2 + 0.5, ny / 2 + 0.5
    elif nx % 2 == 1:
        crpixx, crpixy = nx / 2 + 1, ny / 2 + 1
    crpixs = [crpixx, crpixy, 1, 1, 1, 1]
    for i in range(len(beam.shape)):
        ii = str(i + 1)
        hdr["CTYPE" + ii] = ctypes[i]
        hdr["CRPIX" + ii] = crpixs[i]
        hdr["CRVAL" + ii] = crvals[i]
        hdr["CDELT" + ii] = cdelts[i]
        hdr["CUNIT" + ii] = cunits[i]
    hdr["TELESCOP"] = "MeerKAT"
    hdr["DATE"] = time.ctime()

    # Write real and imag parts of data
    hdu = fits.PrimaryHDU(beam, header=hdr)
    hdu.writeto(filename, overwrite=True)


def write_fits_eight(data, freqs, diameter, prefix):
    C = ["x", "y"]
    if len(data.shape) == 4:
        data = np.expand_dims(data, axis=2)
    for i in range(2):
        for j in range(2):
            filename = f"{prefix}_{C[i]}{C[j]}"
            write_fits_cube(
                data[:, i, j, :, :].real, freqs, diameter, filename + "_re.fits"
            )
            write_fits_cube(
                data[:, i, j, :, :].imag, freqs, diameter, filename + "_im.fits"
            )


def write_fits_single(beam, freqs, diameter, filename):
    data = np.zeros((2, *beam.shape))
    data[0, ...] = beam.real
    data[1, ...] = beam.imag

    # Create header
    hdr = fits.Header()
    diam = float(diameter)
    ctypes = ["X", "Y", "FEED1", "FEED2", "PART"]
    crvals = [0.0, 0.0, 0, 0, 0]
    cdelts = [diam / beam.shape[-2], diam / beam.shape[-1], 1, 1, 1]
    cunits = ["deg", "deg", "", "", ""]
    nx, ny = beam.shape[-2], beam.shape[-1]
    if nx % 2 == 0:
        crpixx, crpixy = nx / 2 + 0.5, ny / 2 + 0.5
    elif nx % 2 == 1:
        crpixx, crpixy = nx / 2 + 1, ny / 2 + 1
    crpixs = [crpixx, crpixy, 1, 1, 1]
    for i in range(len(data.shape)):
        ii = str(i + 1)
        hdr["CTYPE" + ii] = ctypes[i]
        hdr["CRPIX" + ii] = crpixs[i]
        hdr["CRVAL" + ii] = crvals[i]
        hdr["CDELT" + ii] = cdelts[i]
        hdr["CUNIT" + ii] = cunits[i]
    hdr["TELESCOP"] = "MeerKAT"
    hdr["DATE"] = time.ctime()

    # Write real and imag parts of data
    hdu = fits.PrimaryHDU(data, header=hdr)
    hdu.writeto(filename + ".fits", overwrite=True)


# MeerKAT L-band: 857 channels starting at 856 MHz
MEERKAT_LBAND_FREQS = np.arange(857) + 856


def freq_to_idx(freqs=None, sb=(0, 1300)):
    # convert a frequency to an index, default is MeerKAT L-band specifications
    if freqs is None:
        freqs = MEERKAT_LBAND_FREQS
    start = (np.abs(freqs - sb[0])).argmin()
    end = (np.abs(freqs - sb[1])).argmin()
    return start, end
