import scipy.io

def read_joints(action, video, path="/home/linxin67/scratch/JHMDB/"):
    os.chdir(path)
    mat = scipy.io.loadmat(f'{path}{action}/{video}/joint_positions.mat')
    return mat