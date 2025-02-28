# -*- coding: UTF8 -*-
"""
Provides algorithms for time synchronization.
author: Michael Grupp

This file is part of evo (github.com/MichaelGrupp/evo).

evo is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

evo is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with evo.  If not, see <http://www.gnu.org/licenses/>.
"""

import copy
import logging
import typing

import numpy as np

from evo import EvoException
from evo.core.trajectory import PoseTrajectory3D

logger = logging.getLogger(__name__)


class SyncException(EvoException):
    pass


MatchingIndices = typing.Tuple[typing.List[int], typing.List[int]]
TrajectoryPair = typing.Tuple[PoseTrajectory3D, PoseTrajectory3D]


def matching_time_indices(stamps_1: np.ndarray, stamps_2: np.ndarray,
                          max_diff: float = 0.01,
                          offset_2: float = 0.0) -> MatchingIndices:
    """
    Searches for the best matching timestamps of two lists of timestamps
    and returns the list indices of the best matches.
    :param stamps_1: first vector of timestamps (numpy array)
    :param stamps_2: second vector of timestamps (numpy array)
    :param max_diff: max. allowed absolute time difference
    :param offset_2: optional time offset to be applied to stamps_2
    :return: 2 lists of the matching timestamp indices (stamps_1, stamps_2)
    """
    matching_indices_1 = []
    matching_indices_2 = []
    stamps_2 = copy.deepcopy(stamps_2)
    stamps_2 += offset_2
    for index_1, stamp_1 in enumerate(stamps_1):
        diffs = np.abs(stamps_2 - stamp_1)
        index_2 = int(np.argmin(diffs))
        if diffs[index_2] <= max_diff:
            matching_indices_1.append(index_1)
            matching_indices_2.append(index_2)
    return matching_indices_1, matching_indices_2


def interpol_ndarray(t0: float, t1: float, xq0: np.ndarray, xq1: np.ndarray,
                   t: float) -> np.ndarray:
    logger.info("interpol_ndarray with t0: {}, t: {}, t1: {}".format(t0, t, t1))
    logger.info(" x0: {}, x1: {}".format(xq0, xq1))
    logger.info(" -> {}".format(xq0 + (xq1 - xq0) * (t - t0)/(t1 - t0)))
    return xq0 + (xq1 - xq0) * (t - t0)/(t1 - t0)


def interpol_at(traj: PoseTrajectory3D, stamps: np.ndarray) -> PoseTrajectory3D:
    """
    Interpolate a trajectory for a given timestamp and return the interpolated
    position and orientation.
    :param traj: The reference trajectory to interpolate from. Must be sorted.
    :param stamps: The timestamps. Must be sorted.
    """
    is_sorted = lambda a: np.all(a[:-1] <= a[1:])
    if not is_sorted(stamps):
        logger.error("Requested timestamps must be sorted!")
    if not is_sorted(traj.timestamps):
        logger.error("Reference timestamps must be sorted!")

    prev_exists = False
    t0 = 0
    x0 = 0
    q0 = 0
    i_stamps = 0
    t = stamps[i_stamps]

    # Output:
    dropped_i = []
    out_stamps = np.empty([stamps.size, 1])
    out_xyz = np.empty([stamps.size, 3])
    out_quats = np.empty([stamps.size, 4])

    # For each ref_pose in traj:
    for i1, t1 in enumerate(traj.timestamps):
        x1 = traj.positions_xyz[i1]
        q1 = traj.orientations_quat_wxyz[i1]

        # Advance i_stamps until t = stamps(i_stamps) > ref_pose.ts
        while t < t1:
            # For each t store interpolated pose at (t - t0)
            out_stamps[i_stamps] = t
            if prev_exists:
                out_xyz[i_stamps] = interpol_ndarray(t0, t1, x0, x1, t)
                out_quats[i_stamps] = interpol_ndarray(t0, t1, q0, q1, t)
            else:
                logger.warn("Dropping requested timestamp {} earlier than reference".format(t))
                dropped_i.append(i_stamps)

            # Get the next stamp:
            i_stamps += 1
            if i_stamps >= len(stamps):
                logger.warn("end")
                break
            t = stamps[i_stamps]

        # Store previous ref_pose in t0, x0, q0
        t0 = t1
        x0 = x1
        q0 = q1
        prev_exists = True

    if i_stamps < len(stamps):
        for i, t in enumerate(stamps[i_stamps+1:]):
            logger.warn("Dropping requested timestamp {} later than reference".format(t))
            dropped_i.append(i_stamps)

    # Remove all dropped stamps from the ndarray.
    # Remove in reverse in order to not change indices for later removal steps:

    dropped_i.reverse()
    for i in enumerate dropped_i:
        del out_xyz[i]
        del out_quats[i]
        del out_stamps[i]
    
    return PoseTrajectory3D(out_xyz, out_quats, out_stamps)


def associate_trajectories(
        traj_1: PoseTrajectory3D, traj_2: PoseTrajectory3D,
        max_diff: float = 0.01, offset_2: float = 0.0,
        first_name: str = "first trajectory",
        snd_name: str = "second trajectory") -> TrajectoryPair:
    """
    Synchronizes two trajectories by matching their timestamps.
    :param traj_1: trajectory.PoseTrajectory3D object of first trajectory
    :param traj_2: trajectory.PoseTrajectory3D object of second trajectory
    :param max_diff: max. allowed absolute time difference for associating
    :param offset_2: optional time offset of second trajectory
    :param first_name: name of first trajectory for verbose logging
    :param snd_name: name of second trajectory for verbose/debug logging
    :return: traj_1, traj_2 (synchronized)
    """
    if not isinstance(traj_1, PoseTrajectory3D) \
        or not isinstance(traj_2, PoseTrajectory3D):
        raise SyncException("trajectories must be PoseTrajectory3D objects")

    time_1 = traj_1.timestamps[-1] - traj_1.timestamps[0]
    time_2 = traj_2.timestamps[-1] - traj_2.timestamps[0]
    snd_longer = time_2 > time_1
    traj_long = copy.deepcopy(traj_2) if snd_longer else copy.deepcopy(traj_1)
    traj_short = copy.deepcopy(traj_1) if snd_longer else copy.deepcopy(traj_2)

    interpolated = interpol_at(traj_long, traj_short.timestamps)

    traj_1 = traj_short if snd_longer else interpolated
    traj_2 = interpolated if snd_longer else traj_short

    return traj_1, traj_2
