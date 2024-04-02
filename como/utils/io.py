from como.geometry.lie_algebra import pose_to_tq


def save_traj(filename, timestamps, poses):
    file = open(filename, "w")
    for i in range(poses.shape[0]):
        timestamp = timestamps[i]
        tq = pose_to_tq(poses[i, :, :])
        # timestamp tx ty tz qx qy qz qw
        # line = f"{timestamp, t[0], t[1], t[2], q[1], q[2], q[3], q[0]
        line = "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n" % (
            timestamp,
            tq[0],
            tq[1],
            tq[2],
            tq[3],
            tq[4],
            tq[5],
            tq[6],
        )
        file.write(line)

    file.close()
