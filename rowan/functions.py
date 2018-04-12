# Copyright (c) 2018 The Regents of the University of Michigan
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
R"""Submodule containing all standard functions"""
from __future__ import division, print_function, absolute_import

import numpy as np


def exp(q):
    R"""Computes the natural exponential function :math:`e^q`.

    The exponential of a quaternion in terms of its scalar and vector parts
    :math:`q = a + \boldsymbol{v}` is defined by exponential power series:
    formula :math:`e^x = \sum_{k=0}^{\infty} \frac{x^k}{k!}` as follows:

    .. math::
        \begin{align}
            e^q &= e^{a+v} \\
                &= e^a \left(\sum_{k=0}^{\infty} \frac{v^k}{k!} \right) \\
                &= e^a \left(\cos \lvert \lvert \boldsymbol{v} \rvert \rvert +
                    \frac{\boldsymbol{v}}{\lvert \lvert \boldsymbol{v} \rvert
                    \rvert} \sin \lvert \lvert \boldsymbol{v} \rvert \rvert
                    \right)
        \end{align}

    Args:
        q ((...,4) np.array): Quaternions

    Returns:
        Array of shape (...) containing exponentials of q

    Example::

        q_exp = exp(q)
    """
    # Ensure compatibility for numpy < 1.13; older numpy fail with
    # the fancy indexing used below when array is 1d
    q = np.asarray(q)
    if len(q.shape) == 1:
        flat = True
        q = np.atleast_2d(q)
    else:
        flat = False

    expo = np.empty(q.shape)
    norms = np.linalg.norm(q[..., 1:], axis=-1)
    e = np.exp(q[..., 0])
    expo[..., 0] = e * np.cos(norms)
    norm_zero = np.isclose(norms, 0)
    not_zero = np.logical_not(norm_zero)
    if np.any(not_zero):
        expo[not_zero, 1:] = e[not_zero, np.newaxis] * (
                q[not_zero, 1:]/norms[not_zero, np.newaxis]
                ) * np.sin(norms)[not_zero, np.newaxis]
        if np.any(norm_zero):
            expo[norm_zero, 1:] = 0
    else:
        expo[..., 1:] = 0

    if flat:
        return expo.squeeze()
    else:
        return expo


def expb(q, b):
    R"""Computes the exponential function :math:`b^q`.

    We define the exponential of a quaternion to an arbitrary base relative
    to the exponential function :math:`e^q` using the change of base
    formula as follows:

    .. math::
        \begin{align}
            b^q &= y \\
            q &= \log_b y  = \frac{\ln y}{\ln b}\\
            y &= e^{q\ln b}
        \end{align}

    Args:
        q ((...,4) np.array): Quaternions

    Returns:
        Array of shape (...) containing exponentials of q

    Example::

        q_exp = exp(q, 2)
    """
    return exp(q*np.log(b))


def exp10(q):
    R"""Computes the exponential function :math:`10^q`.

    Wrapper around :py:func:`expb`.

    Args:
        q ((...,4) np.array): Quaternions

    Returns:
        Array of shape (...) containing exponentials of q

    Example::

        q_exp = exp(q, 2)
    """
    return expb(q, 10)


def log(q):
    R"""Computes the quaternion natural logarithm.

    The natural of a quaternion in terms of its scalar and vector parts
    :math:`q = a + \boldsymbol{v}` is defined by inverting the exponential
    formula (see :py:func:`exp`), and is defined by the formula
    :math:` \frac{x^k}{k!}` as follows:

    .. math::
        \begin{equation}
            \ln(q) = \ln\lvert\lvert q \rvert\rvert +
                    \frac{\boldsymbol{v}}{\lvert\lvert \boldsymbol{v}
                    \rvert\rvert} \arccos\left(\frac{a}{q}\right)
        \end{equation}

    Args:
        q ((...,4) np.array): Quaternions

    Returns:
        Array of shape (...) containing logarithms of q

    Example::

        ln_q = log(q)
    """
    # Ensure compatibility for numpy < 1.13; older numpy fail with
    # the fancy indexing used below when array is 1d
    q = np.asarray(q)
    if len(q.shape) == 1:
        flat = True
        q = np.atleast_2d(q)
    else:
        flat = False

    log = np.empty(q.shape)

    # We need all the norms to avoid divide by zeros later.
    # Can also use these to minimize the amount of work done.
    q_norms = norm(q)
    q_norm_zero = np.isclose(q_norms, 0)
    q_not_zero = np.logical_not(q_norm_zero)
    v_norms = np.linalg.norm(q[..., 1:], axis=-1)
    v_norm_zero = np.isclose(v_norms, 0)
    v_not_zero = np.logical_not(v_norm_zero)

    if np.any(q_not_zero):
        if np.any(q_norm_zero):
            log[q_norm_zero, 0] = -np.inf
        log[q_not_zero, 0] = np.log(q_norms[q_not_zero])
    else:
        log[..., 0] = -np.inf

    if np.any(v_not_zero):
        prefactor = np.empty(q[v_not_zero, 1:].shape)
        prefactor = q[v_not_zero, 1:]/v_norms[
                v_not_zero, np.newaxis]

        inv_cos = np.empty(v_norms[v_not_zero].shape)
        inv_cos = np.arccos(q[v_not_zero, 0]/q_norms[v_not_zero])

        if np.any(v_norm_zero):
            log[v_norm_zero, 1:] = 0
        log[v_not_zero, 1:] = prefactor * inv_cos[..., np.newaxis]
    else:
        log[..., 1:] = 0

    if flat:
        return log.squeeze()
    else:
        return log


def logb(q, b):
    R"""Computes the quaternion logarithm to some base b.

    The quaternion logarithm for arbitrary bases is defined using the
    standard change of basis formula relative to the natural logarithm.

    .. math::
        \begin{align}
            \log_b q &= y \\
            q &= b^y \\
            \ln q &= y \ln b \\
            y &= \log_b q = \frac{\ln q}{\ln b}
        \end{align}

    Args:
        q ((...,4) np.array): Quaternions
        n ((...) np.array): Scalars to use as log bases

    Returns:
        Array of shape (...) containing logarithms of q

    Example::

        log_q = log(q, 2)
    """
    q = np.asarray(q)
    return log(q)/np.log(b)


def log10(q):
    R"""Computes the quaternion logarithm base 10.

    Wrapper around :py:func:`logb`.

    Args:
        q ((...,4) np.array): Quaternions

    Returns:
        Array of shape (...) containing logarithms of q

    Example::

        log_q = log(q, 2)
    """
    q = np.asarray(q)
    return logb(q, 10)


def power(q, n):
    # TODO: Write polar decomposition function #noqa
    R"""Computes the power of a quaternion :math:`q^n`.

    Quaternions raised to a scalar power are defined according to the polar
    decomposition angle :math:`\theta` and vector :math:`\hat{u}`:
    :math:`q^n = \lvert\lvert q \rvert\rvert^n \left( \cos(n\theta) + \hat{u}
    \sin(n\theta)\right)`. However, this can be computed
    more efficiently by noting that :math:`q^n = \exp(n \ln(q))`.

    Args:
        q ((...,4) np.array): Quaternions.
        n ((...) np.arrray): Scalars to exponentiate quaternions with.

    Returns:
        Array of shape (...) containing  of q

    Example::

        q_n = pow(q^n)
    """
    q = np.asarray(q)
    # Note that we follow the convention that 0^0 = 1
    if n == 0:
        return np.broadcast_to(np.array([1, 0, 0, 0]), q.shape)
    else:
        return exp(n*log(q))


def conjugate(q):
    R"""Conjugates an array of quaternions

    Args:
        q ((...,4) np.array): Array of quaternions

    Returns:
        An array containing the conjugates of q

    Example::

        q_star = conjugate(q)
    """
    # Don't use asarray to avoid modifying in place
    conjugate = np.array(q)
    conjugate[..., 1:] *= -1
    return conjugate


def inverse(q):
    R"""Computes the inverse of an array of quaternions

    Args:
        q ((...,4) np.array): Array of quaternions

    Returns:
        An array containing the inverses of q

    Example::

        q_inv = inverse(q)
    """
    q = np.asarray(q)

    if len(q.shape) == 1:
        flat = True
        inverses = np.array(np.atleast_2d(q))
    else:
        flat = False
        inverses = np.array(q)

    normsq = norm(inverses)**2
    if np.any(normsq):
        inverses[..., 1:] *= -1
        # Would like to do this in place, but can't guarantee type safety
        inverses[normsq > 0] = inverses[normsq > 0]/normsq[
                normsq > 0, np.newaxis]

    if flat:
        return inverses.squeeze()
    else:
        return inverses


def multiply(qi, qj):
    R"""Multiplies two arrays of quaternions

    Note that quaternion multiplication is generally non-commutative.

    Args:
        qi ((...,4) np.array): First set of quaternions
        qj ((...,4) np.array): Second set of quaternions

    Returns:
        An array containing the products of row i of qi
        with column j of qj

    Example::

        qi = np.array([[1, 0, 0, 0]])
        qj = np.array([[1, 0, 0, 0]])
        prod = multiply(qi, qj)
    """
    qi = np.asarray(qi)
    qj = np.asarray(qj)

    output = np.empty(np.broadcast(qi, qj).shape)

    output[..., 0] = qi[..., 0] * qj[..., 0] - \
        np.sum(qi[..., 1:] * qj[..., 1:], axis=-1)
    output[..., 1:] = (qi[..., 0, np.newaxis] * qj[..., 1:] +
                       qj[..., 0, np.newaxis] * qi[..., 1:] +
                       np.cross(qi[..., 1:], qj[..., 1:]))
    return output


def divide(qi, qj):
    R"""Divides two arrays of quaternions

    Division is non-commutative; this function returns
    :math:`q_i q_j^{-1}`.

    Args:
        qi ((...,4) np.array): Dividend quaternion
        qj ((...,4) np.array): Divisors quaternions

    Returns:
        An array containing the quotients of row i of qi
        with column j of qj

    Example::

        qi = np.array([[1, 0, 0, 0]])
        qj = np.array([[1, 0, 0, 0]])
        prod = divide(qi, qj)
    """
    return multiply(qi, inverse(qj))


def norm(q):
    R"""Compute the quaternion norm

    Args:
        q ((...,4) np.array): Quaternions to find norms for

    Returns:
        An array containing the norms for qi in q

    Example::

        q = np.random.rand(10, 4)
        norms = norm(q)
    """
    q = np.asarray(q)
    return np.linalg.norm(q, axis=-1)


def normalize(q):
    R"""Normalize quaternions

    Args:
        q ((...,4) np.array): Array of quaternions to normalize

    Returns:
        An array containing the unit quaternions q/norm(q)

    Example::

        q = np.random.rand(10, 4)
        u = normalize(q)
    """
    q = np.asarray(q)
    norms = norm(q)
    return q / norms[..., np.newaxis]


def from_mirror_plane(x, y, z):
    R"""Generate quaternions from mirror plane equations.

    Reflection quaternions can be constructed of the from
    :math:`(0, x, y, z)`, *i.e.* with zero real component. The vector
    :math:`(x, y, z)` is the normal to the mirror plane.

    Args:
        x ((...) np.array): First planar component
        y ((...) np.array): Second planar component
        z ((...) np.array): Third planar component

    Returns:
        An array of quaternions corresponding to the provided reflections.

    Example::

        plane = (1, 2, 3)
        quat_ref = from_mirror_plane(*plane)
    """
    x, y, z = np.broadcast_arrays(x, y, z)
    q = np.empty(x.shape + (4,))
    q[..., 0] = 0
    q[..., 1] = x
    q[..., 2] = y
    q[..., 3] = z

    return q


def _promote_vec(v):
    R"""Helper function to promote vectors to their quaternion representation"""
    return np.concatenate((np.zeros(v.shape[:-1] + (1,)), v), axis=-1)


def reflect(q, v):
    R"""Reflect a list of vectors by a corresponding set of quaternions

    For help constructing a mirror plane, see :py:func:`from_mirror_plane`.

    Args:
        q ((...,4) np.array): Quaternions to use for reflection
        v ((...,3) np.array): Vectors to reflect.

    Returns:
        An array of the vectors in v reflected by q

    Example::

        q = np.random.rand(1, 4)
        v = np.random.rand(1, 3)
        v_rot = rotate(q, v)
    """
    q = np.asarray(q)
    v = np.asarray(v)

    if not np.allclose(norm(q), 1):
        raise ValueError("Reflection quaternions must have unit norm")

    # Convert vector to quaternion representation
    quat_v = _promote_vec(v)
    return multiply(q, multiply(quat_v, q))[..., 1:]


def rotate(q, v):
    R"""Rotate a list of vectors by a corresponding set of quaternions

    Args:
        q ((...,4) np.array): Quaternions to rotate by.
        v ((...,3) np.array): Vectors to rotate.

    Returns:
        An array of the vectors in v rotated by q

    Example::

        q = np.random.rand(1, 4)
        v = np.random.rand(1, 3)
        v_rot = rotate(q, v)
    """
    q = np.asarray(q)
    v = np.asarray(v)

    if not np.allclose(norm(q), 1):
        raise ValueError("Rotation quaternions must have unit norm")

    # Convert vector to quaternion representation
    quat_v = _promote_vec(v)
    return multiply(q, multiply(quat_v, conjugate(q)))[..., 1:]


def _normalize_vec(v):
    R"""Helper function to normalize vectors"""
    v = np.asarray(v)
    norms = np.linalg.norm(v, axis=-1)
    return v / norms[..., np.newaxis]


def _vector_bisector(v1, v2):
    R"""Find the vector bisecting two vectors

    Args:
        v1 ((...,3) np.array): First vector
        v2 ((...,3) np.array): Second vector

    Returns:
        The vector that bisects the angle between v1 and v2
    """

    return _normalize_vec(_normalize_vec(v1) + _normalize_vec(v2))


def vector_vector_rotation(v1, v2):
    R"""Find the quaternion to rotate one vector onto another

    Args:
        v1 ((...,3) np.array): Vector to rotate
        v2 ((...,3) np.array): Desired vector

    Returns:
        Array (..., 4) of quaternions that rotate v1 onto v2.
    """
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    return from_axis_angle(_vector_bisector(v1, v2), np.pi)


def from_euler(alpha, beta, gamma, convention='zyx',
               axis_type='intrinsic'):
    R"""Convert Euler angles to quaternions

    For generality, the rotations are computed by composing a sequence of
    quaternions corresponding to axis-angle rotations. While more efficient
    implementations are possible, this method was chosen to prioritize
    flexibility since it works for essentially arbitrary Euler angles as
    long as intrinsic and extrinsic rotations are not intermixed.

    Args:
        alpha ((...) np.array): Array of :math:`\alpha` values
        beta ((...) np.array): Array of :math:`\beta` values
        gamma ((...) np.array): Array of :math:`\gamma` values
        convention (str): One of the 12 valid conventions xzx, xyx,
            yxy, yzy, zyz, zxz, xzy, xyz, yxz, yzx, zyx, zxy
        axes (str): Whether to use extrinsic or intrinsic rotations

    Returns:
        An array containing the converted quaternions

    Example::

        rands = np.random.rand(100, 3)
        alpha, beta, gamma = rands.T
        ql.from_euler(alpha, beta, gamma)
    """
    angles = np.broadcast_arrays(alpha, beta, gamma)

    convention = convention.lower()

    if len(convention) > 3 or (set(convention) - set('xyz')):
        raise ValueError("All acceptable conventions must be 3 \
character strings composed only of x, y, and z")

    basis_axes = {
        'x': np.array([1, 0, 0]),
        'y': np.array([0, 1, 0]),
        'z': np.array([0, 0, 1]),
    }
    # Temporary method to ensure shapes conform
    for ax, vec in basis_axes.items():
        basis_axes[ax] = np.broadcast_to(
            vec,
            angles[0].shape + (vec.shape[-1],)
        )

    # Split by convention, the easiest
    rotations = []
    if axis_type == 'extrinsic':
        # Loop over the axes and add each rotation
        for i, char in enumerate(convention):
            ax = basis_axes[char]
            rotations.append(from_axis_angle(ax, angles[i]))
    elif axis_type == 'intrinsic':
        for i, char in enumerate(convention):
            ax = basis_axes[char]
            rotations.append(from_axis_angle(ax, angles[i]))
            # Rotate the bases as well
            for key, value in basis_axes.items():
                basis_axes[key] = rotate(
                    rotations[-1],
                    value
                )
    else:
        raise ValueError("Only valid axis_types are intrinsic and extrinsic")

    # Compose the total rotation
    final_rotation = np.broadcast_to(
        np.array([1, 0, 0, 0]),
        rotations[0].shape
    )
    for q in rotations:
        final_rotation = multiply(q, final_rotation)

    return final_rotation


def to_euler(q, convention='zyx', axis_type='intrinsic'):
    R"""Convert quaternions to Euler angles

    Euler angles are returned in the sequence provided, so in, *e.g.*,
    the default case ('zyx'), the angles returned are for a rotation
    :math:`Z(\alpha) Y(\beta) X(\gamma)`.

    .. note::

        In all cases, the :math:`\alpha` and :math:`\gamma` angles are
        between :math:`\pm \pi`. For proper Euler angles, :math:`\beta`
        is between :math:`0` and :math:`pi` degrees. For Tait-Bryan
        angles, :math:`\beta` lies between :math:`\pm\pi/2`.

    For simplicity, quaternions are converted to matrices, which are
    then converted to their Euler angle representations. All equations
    for rotations are derived by considering compositions of the three
    elemental rotations about the three Cartesian axes:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        R_x(\theta)  =& \left(\begin{array}{ccc}
                            1   & 0             & 0 \\
                            0   & \cos \theta    & -\sin \theta \\
                            0   & \sin \theta    & \cos \theta    \\
                         \end{array}\right)\\
        R_y(\theta)  =& \left(\begin{array}{ccc}
                            \cos \theta   & 0        & \sin \theta \\
                            0            & 1        &  0\\
                            -\sin \theta  & 1        & \cos \theta    \\
                         \end{array}\right)\\
        R_z(\theta)  =& \left(\begin{array}{ccc}
                            \cos \theta  & -\sin \theta   & 0 \\
                            \sin \theta  & \cos \theta    & 0 \\
                            0           & 0             & 1 \\
                         \end{array}\right)\\
        \end{eqnarray*}

    Extrinsic rotations are represented by matrix multiplications in
    the proper order, so :math:`z-y-x` is represented by the
    multiplication :math:`XYZ` so that the system is rotated first
    about :math:`Z`, then about :math:`y`, then finally :math:`X`.
    For intrinsic rotations, the order of rotations is reversed,
    meaning that it matches the order in which the matrices actually
    appear *i.e.* the :math:`z-y'-x''` convention (yaw, pitch, roll)
    corresponds to the multiplication of matrices :math:`ZYX`.
    For proof of the relationship between intrinsic and extrinsic
    rotations, see the `Wikipedia page on Davenport chained rotations
    <https://en.wikipedia.org/wiki/Davenport_chained_rotations>`_.

    For more information, see the Wikipedia page for
    `Euler angles <https://en.wikipedia.org/wiki/Euler_angles>`_
    (specifically the section on converting between representations).

    Args:
        q ((...,4) np.array): Quaternions to transform
        convention (str): One of the 6 valid conventions zxz,
            xyx, yzy, zyz, xzx, yxy
        axes (str): Whether to use extrinsic or intrinsic

    Returns:
        An array with Euler angles :math:`(\alpha, \beta, \gamma)`
        as the last dimension (in radians)

    Example::

        rands = np.random.rand(100, 3)
        alpha, beta, gamma = rands.T
        ql.from_euler(alpha, beta, gamma)
        alpha_return, beta_return, gamma_return = ql.to_euler(full)

    """
    q = np.asarray(q)

    try:
        mats = to_matrix(q)
    except ValueError:
        raise ValueError(
            "Not all quaternions in q are unit quaternions.")
    except: # noqa E722
        raise

    if axis_type == 'intrinsic':
        # Have to hardcode the different possibilites.
        # Classical Euler angles
        if convention == 'xzx':
            alpha = np.arctan2(mats[..., 2, 0], mats[..., 1, 0])
            beta = np.arccos(mats[..., 0, 0])
            gamma = np.arctan2(mats[..., 0, 2], -mats[..., 0, 1])
        elif convention == 'xyx':
            alpha = np.arctan2(mats[..., 1, 0], -mats[..., 2, 0])
            beta = np.arccos(mats[..., 0, 0])
            gamma = np.arctan2(mats[..., 0, 1], mats[..., 0, 2])
        elif convention == 'yxy':
            alpha = np.arctan2(mats[..., 0, 1], mats[..., 2, 1])
            beta = np.arccos(mats[..., 1, 1])
            gamma = np.arctan2(mats[..., 1, 0], -mats[..., 1, 2])
        elif convention == 'yzy':
            alpha = np.arctan2(mats[..., 2, 1], -mats[..., 0, 1])
            beta = np.arccos(mats[..., 1, 1])
            gamma = np.arctan2(mats[..., 1, 2], mats[..., 1, 0])
        elif convention == 'zyz':
            alpha = np.arctan2(mats[..., 1, 2], mats[..., 0, 2])
            beta = np.arccos(mats[..., 2, 2])
            gamma = np.arctan2(mats[..., 2, 1], -mats[..., 2, 0])
        elif convention == 'zxz':
            alpha = np.arctan2(mats[..., 0, 2], -mats[..., 1, 2])
            beta = np.arccos(mats[..., 2, 2])
            gamma = np.arctan2(mats[..., 2, 0], mats[..., 2, 1])
        # Tait-Bryan angles
        elif convention == 'xzy':
            alpha = np.arctan2(mats[..., 2, 1], mats[..., 1, 1])
            beta = np.arcsin(-mats[..., 0, 1])
            gamma = np.arctan2(mats[..., 0, 2], mats[..., 0, 0])
        elif convention == 'xyz':
            alpha = np.arctan2(-mats[..., 1, 2], mats[..., 2, 2])
            beta = np.arcsin(mats[..., 0, 2])
            gamma = np.arctan2(-mats[..., 0, 1], mats[..., 0, 0])
        elif convention == 'yxz':
            alpha = np.arctan2(mats[..., 0, 2], mats[..., 2, 2])
            beta = np.arcsin(-mats[..., 1, 2])
            gamma = np.arctan2(mats[..., 1, 0], mats[..., 1, 1])
        elif convention == 'yzx':
            alpha = np.arctan2(-mats[..., 2, 0], mats[..., 0, 0])
            beta = np.arcsin(mats[..., 1, 0])
            gamma = np.arctan2(-mats[..., 1, 2], mats[..., 1, 1])
        elif convention == 'zyx':
            alpha = np.arctan2(mats[..., 1, 0], mats[..., 0, 0])
            beta = np.arcsin(-mats[..., 2, 0])
            gamma = np.arctan2(mats[..., 2, 1], mats[..., 2, 2])
        elif convention == 'zxy':
            alpha = np.arctan2(-mats[..., 0, 1], mats[..., 1, 1])
            beta = np.arcsin(mats[..., 2, 1])
            gamma = np.arctan2(-mats[..., 2, 0], mats[..., 2, 2])
        else:
            raise ValueError("Unknown convention selected!")
    elif axis_type == 'extrinsic':
        # For these, the matrix must be constructed in reverse order
        # e.g. Z(\alpha)Y'(\beta)Z''(\gamma) (where primes denote the
        # rotated frames) becomes the extrinsic rotation
        # Z(\gamma)Y(\beta)Z(\alpha).

        # Classical Euler angles
        if convention == 'xzx':
            alpha = np.arctan2(mats[..., 0, 2], -mats[..., 0, 1])
            beta = np.arccos(mats[..., 0, 0])
            gamma = np.arctan2(mats[..., 2, 0], mats[..., 1, 0])
        elif convention == 'xyx':
            alpha = np.arctan2(mats[..., 0, 1], mats[..., 0, 2])
            beta = np.arccos(mats[..., 0, 0])
            gamma = np.arctan2(mats[..., 1, 0], -mats[..., 2, 0])
        elif convention == 'yxy':
            alpha = np.arctan2(mats[..., 1, 0], -mats[..., 1, 2])
            beta = np.arccos(mats[..., 1, 1])
            gamma = np.arctan2(mats[..., 0, 1], mats[..., 2, 1])
        elif convention == 'yzy':
            alpha = np.arctan2(mats[..., 1, 2], mats[..., 1, 0])
            beta = np.arccos(mats[..., 1, 1])
            gamma = np.arctan2(mats[..., 2, 1], -mats[..., 0, 1])
        elif convention == 'zyz':
            alpha = np.arctan2(mats[..., 2, 1], -mats[..., 2, 0])
            beta = np.arccos(mats[..., 2, 2])
            gamma = np.arctan2(mats[..., 1, 2], mats[..., 0, 2])
        elif convention == 'zxz':
            alpha = np.arctan2(mats[..., 2, 0], mats[..., 2, 1])
            beta = np.arccos(mats[..., 2, 2])
            gamma = np.arctan2(mats[..., 0, 2], -mats[..., 1, 2])
        # Tait-Bryan angles
        elif convention == 'xzy':
            alpha = np.arctan2(-mats[..., 1, 2], mats[..., 1, 1])
            beta = np.arcsin(mats[..., 1, 0])
            gamma = np.arctan2(-mats[..., 2, 0], mats[..., 0, 0])
        elif convention == 'xyz':
            alpha = np.arctan2(mats[..., 2, 1], mats[..., 2, 2])
            beta = np.arcsin(-mats[..., 2, 0])
            gamma = np.arctan2(mats[..., 1, 0], mats[..., 0, 0])
        elif convention == 'yxz':
            alpha = np.arctan2(-mats[..., 2, 0], mats[..., 2, 2])
            beta = np.arcsin(mats[..., 2, 1])
            gamma = np.arctan2(-mats[..., 0, 1], mats[..., 1, 1])
        elif convention == 'yzx':
            alpha = np.arctan2(mats[..., 0, 2], mats[..., 0, 0])
            beta = np.arcsin(-mats[..., 0, 1])
            gamma = np.arctan2(mats[..., 2, 1], mats[..., 1, 1])
        elif convention == 'zyx':
            alpha = np.arctan2(-mats[..., 0, 1], mats[..., 0, 0])
            beta = np.arcsin(mats[..., 0, 2])
            gamma = np.arctan2(-mats[..., 1, 2], mats[..., 2, 2])
        elif convention == 'zxy':
            alpha = np.arctan2(mats[..., 1, 0], mats[..., 1, 1])
            beta = np.arcsin(-mats[..., 1, 2])
            gamma = np.arctan2(mats[..., 0, 2], mats[..., 2, 2])
        else:
            raise ValueError("Unknown convention selected!")
    else:
        raise ValueError("The axis type must be either extrinsic or intrinsic")

    return np.stack((alpha, beta, gamma), axis=-1)


def from_matrix(mat, require_orthogonal=True):
    R"""Convert the rotation matrices mat to quaternions

    Thhis method uses the algorithm described by Bar-Itzhack in [Itzhack00]_.
    The idea is to construct a matrix K whose largest eigenvalue corresponds
    to the desired quaternion. One of the strengths of the algorithm is that
    for nonorthogonal matrices it gives the closest quaternion representation
    rather than failing outright.

    .. [Itzhack00] Itzhack Y. Bar-Itzhack.  "New Method for Extracting the
        Quaternion from a Rotation Matrix", Journal of Guidance, Control, and
        Dynamics, Vol. 23, No. 6 (2000), pp. 1085-1087
        https://doi.org/10.2514/2.4654

    Args:
        mat ((...,3,3) np.array): An array of rotation matrices

    Returns:
        An array containing the quaternion representations
        of the elements of mat (i.e. the same elements of SO(3))
    """
    mat = np.asarray(mat)
    if require_orthogonal and not np.allclose(np.linalg.det(mat), 1):
        raise ValueError(
            "Not all of your matrices are orthogonal. \
Please ensure that there are no improper rotations. \
If this was intentional, set require_orthogonal to \
False when calling this function.")

    K = np.zeros(mat.shape[:-2] + (4, 4))
    K[..., 0, 0] = mat[..., 0, 0] - mat[..., 1, 1] - mat[..., 2, 2]
    K[..., 0, 1] = mat[..., 1, 0] + mat[..., 0, 1]
    K[..., 0, 2] = mat[..., 2, 0] + mat[..., 0, 2]
    K[..., 0, 3] = mat[..., 1, 2] - mat[..., 2, 1]
    K[..., 1, 0] = mat[..., 1, 0] + mat[..., 0, 1]
    K[..., 1, 1] = mat[..., 1, 1] - mat[..., 0, 0] - mat[..., 2, 2]
    K[..., 1, 2] = mat[..., 2, 1] + mat[..., 1, 2]
    K[..., 1, 3] = mat[..., 2, 0] - mat[..., 0, 2]
    K[..., 2, 0] = mat[..., 2, 0] + mat[..., 0, 2]
    K[..., 2, 1] = mat[..., 2, 1] + mat[..., 1, 2]
    K[..., 2, 2] = mat[..., 2, 2] - mat[..., 0, 0] - mat[..., 1, 1]
    K[..., 2, 3] = mat[..., 0, 1] - mat[..., 1, 0]
    K[..., 3, 0] = mat[..., 1, 2] - mat[..., 2, 1]
    K[..., 3, 1] = mat[..., 2, 0] - mat[..., 0, 2]
    K[..., 3, 2] = mat[..., 0, 1] - mat[..., 1, 0]
    K[..., 3, 3] = mat[..., 0, 0] + mat[..., 1, 1] + mat[..., 2, 2]
    K = K / 3.0

    w, v = np.linalg.eigh(K)
    # The conventions in the paper are very confusing for quaternions in terms
    # of the order of the components
    return np.concatenate(
        (v[..., -1, -1, np.newaxis], -v[..., :-1, -1]), axis=-1)


def to_matrix(q, require_unit=True):
    R"""Convert quaternions into rotation matrices.

    Uses the conversion described on `Wikipedia
    <https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix>`_.

    Args:
        q ((...,4) np.array): An array of quaternions

    Returns:
        The array containing the matrix representations
        of the elements of q (i.e. the same elements of SO(3))
    """
    q = np.asarray(q)

    s = norm(q)
    if np.any(s == 0.0):
        raise ZeroDivisionError(
            "At least one element of q has approximately zero norm")
    elif require_unit and not np.allclose(s, 1.0):
        raise ValueError(
            "Not all quaternions in q are unit quaternions. \
If this was intentional, please set require_unit to False when \
calling this function.")
    m = np.empty(q.shape[:-1] + (3, 3))
    s **= -1.0  # For consistency with Wikipedia notation
    m[..., 0, 0] = 1.0 - 2 * s * (q[..., 2]**2 + q[..., 3]**2)
    m[..., 0, 1] = 2 * (q[..., 1] * q[..., 2] - q[..., 3] * q[..., 0])
    m[..., 0, 2] = 2 * (q[..., 1] * q[..., 3] + q[..., 2] * q[..., 0])
    m[..., 1, 0] = 2 * (q[..., 1] * q[..., 2] + q[..., 3] * q[..., 0])
    m[..., 1, 1] = 1.0 - 2 * (q[..., 1]**2 + q[..., 3]**2)
    m[..., 1, 2] = 2 * (q[..., 2] * q[..., 3] - q[..., 1] * q[..., 0])
    m[..., 2, 0] = 2 * (q[..., 1] * q[..., 3] - q[..., 2] * q[..., 0])
    m[..., 2, 1] = 2 * (q[..., 2] * q[..., 3] + q[..., 1] * q[..., 0])
    m[..., 2, 2] = 1.0 - 2 * (q[..., 1]**2 + q[..., 2]**2)
    return m


def from_axis_angle(axes, angles):
    R"""Find quaternions to rotate a specified angle about a specified axis

    Args:
        axes ((...,3) np.array): An array of vectors (the axes)
        angles (float or (...,1) np.array): An array of angles in radians.
            Will be broadcast to match shape of v as needed

    Returns:
        An array of the desired rotation quaternions

    Example::

        import numpy as np
        axis = np.array([[1, 0, 0]])
        ang = np.pi/3
        quat = about_axis(axis, ang)
    """
    axes = np.asarray(axes)

    # First reshape angles and compute the half angle
    bc = np.broadcast(angles, axes[..., 0])
    angles = np.broadcast_to(angles, bc.shape)[..., np.newaxis]
    axes = np.broadcast_to(axes, bc.shape + (3,))
    ha = angles / 2.0

    # Normalize the vector
    u = _normalize_vec(axes)

    # Compute the components of the quaternions
    scalar_comp = np.cos(ha)
    vec_comp = np.sin(ha) * u

    return np.concatenate((scalar_comp, vec_comp), axis=-1)


def to_axis_angle(q):
    R"""Convert the quaternions in q to axis angle representations

    Args:
        q ((...,4) np.array): An array of quaternions

    Returns:
        A tuple of np.arrays (axes, angles) where axes has
        shape (...,3) and angles has shape (...,1). The
        angles are in radians
    """
    q = np.asarray(q)

    angles = 2*np.atleast_1d(np.arccos(q[..., 0]))
    sines = np.sin(angles/2)
    # Avoid divide by zero issues; these values will not be used
    sines[sines == 0] = 1
    axes = np.where(angles != 0,
                    q[..., 1:]/sines,
                    0)

    return axes, angles


def equal(p, q):
    R"""Check whether two sets of quaternions are equal.

    This function is a simple wrapper that checks array
    equality and then aggregates along the quaternion axis.

    Args:
        p ((...,4) np.array): First set of quaternions
        q ((...,4) np.array): First set of quaternions

    Returns:
        A boolean array of shape (...) indicating equality.
    """
    return np.all(p == q, axis=-1)


def not_equal(p, q):
    R"""Check whether two sets of quaternions are not equal.

    This function is a simple wrapper that checks array
    equality and then aggregates along the quaternion axis.

    Args:
        p ((...,4) np.array): First set of quaternions
        q ((...,4) np.array): First set of quaternions

    Returns:
        A boolean array of shape (...) indicating inequality.
    """
    return np.any(p != q, axis=-1)


def isnan(q):
    R"""Test element-wise for NaN quaternions.

    A quaternion is defined as NaN if any elements are NaN.

    Args:
        q ((...,4) np.array): Quaternions to check

    Returns:
        A boolean array of shape (...) indicating NaN.
    """
    return np.any(np.isnan(q), axis=-1)


def isinf(q):
    R"""Test element-wise for infinite quaternions.

    A quaternion is defined as infinite if any elements are infinite.

    Args:
        q ((...,4) np.array): Quaternions to check

    Returns:
        A boolean array of shape (...) indicating infinite quaternions.
    """
    return np.any(np.isinf(q), axis=-1)


def isfinite(q):
    R"""Test element-wise for finite quaternions.

    A quaternion is defined as finite if all elements are finite.

    Args:
        q ((...,4) np.array): Quaternions to check

    Returns:
        A boolean array of shape (...) indicating finite quaternions.
    """
    return np.all(np.isfinite(q), axis=-1)


def allclose(p, q, **kwargs):
    R"""Check whether two sets of quaternions are all close.

    This is a direct wrapper of the corresponding numpy function.

    Args:
        p ((...,4) np.array): First set of quaternions
        q ((...,4) np.array): First set of quaternions
        **kwargs: Keyword arguments to pass to np.allclose

    Returns:
        Whether or not all quaternions are close
    """
    return np.allclose(p, q, **kwargs)


def isclose(p, q, **kwargs):
    R"""Element-wise check of whether two sets of quaternions close.

    This function is a simple wrapper that checks using the
    corresponding numpy function and then aggregates along
    the quaternion axis.

    Args:
        p ((...,4) np.array): First set of quaternions
        q ((...,4) np.array): First set of quaternions
        **kwargs: Keyword arguments to pass to np.isclose

    Returns:
        A boolean array of shape (...)
    """
    return np.all(np.isclose(p, q, **kwargs), axis=-1)
