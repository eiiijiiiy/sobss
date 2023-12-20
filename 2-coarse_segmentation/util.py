
import numpy as np
import open3d as o3d


def create_box_mesh_from_params(boxes, output_mesh_path):
    num_boxes = boxes.shape[0]
    boxes_coords = np.zeros((num_boxes, 4, 2))

    # right upper
    boxes_coords[:, 0, 0] = boxes[:, 2] \
            + (boxes[:, 5]) * boxes[:, 0] \
            + (boxes[:, 6]/2) * (-boxes[:, 1])
    boxes_coords[:, 0, 1] = boxes[:, 3] \
            + (boxes[:, 5]) * boxes[:, 1] \
            + (boxes[:, 6]/2) * (boxes[:, 0])
    # right lower
    boxes_coords[:, 1, 0] = boxes[:, 2] \
            + (boxes[:, 5]) * boxes[:, 0] \
            - (boxes[:, 6]/2) * (-boxes[:, 1])
    boxes_coords[:, 1, 1] = boxes[:, 3] \
            + (boxes[:, 5]) * boxes[:, 1] \
            - (boxes[:, 6]/2) * (boxes[:, 0])
    # left lower
    boxes_coords[:, 2, 0] = boxes[:, 2] \
            - (boxes[:, 5]) * boxes[:, 0] \
            - (boxes[:, 6]/2) * (-boxes[:, 1])
    boxes_coords[:, 2, 1] = boxes[:, 3] \
            - (boxes[:, 5]) * boxes[:, 1] \
            - (boxes[:, 6]/2) * (boxes[:, 0])
    # left upper
    boxes_coords[:, 3, 0] = boxes[:, 2] \
            - (boxes[:, 5]) * boxes[:, 0] \
            + (boxes[:, 6]/2) * (-boxes[:, 1])
    boxes_coords[:, 3, 1] = boxes[:, 3] \
            - (boxes[:, 5]) * boxes[:, 1] \
            + (boxes[:, 6]/2) * (boxes[:, 0])
    objf = open(output_mesh_path, 'w')

    objf.write('Ka 1.000000 1.000000 1.000000\n')
    objf.write('Kd 1.000000 1.000000 1.000000\n')
    objf.write('Ks 0.000000 0.000000 0.000000\n')
    objf.write('Tr 1.000000\n')
    objf.write('illum 1\n')
    objf.write('Ns 0.000000 1\n')

    for i in range(num_boxes):
        z = boxes[i, 4]
        half_z = boxes[i, 7] / 2

        objf.write("v {:.3f} {:.3f} {:.3f} \n".format(
            boxes_coords[i, 0, 0], boxes_coords[i, 0, 1], z + half_z))
        objf.write("v {:.3f} {:.3f} {:.3f} \n".format(
            boxes_coords[i, 1, 0], boxes_coords[i, 1, 1], z + half_z))
        objf.write("v {:.3f} {:.3f} {:.3f} \n".format(
            boxes_coords[i, 2, 0], boxes_coords[i, 2, 1], z + half_z))
        objf.write("v {:.3f} {:.3f} {:.3f} \n".format(
            boxes_coords[i, 3, 0], boxes_coords[i, 3, 1], z + half_z))

        objf.write("v {:.3f} {:.3f} {:.3f} \n".format(
            boxes_coords[i, 0, 0], boxes_coords[i, 0, 1], z - half_z))
        objf.write("v {:.3f} {:.3f} {:.3f} \n".format(
            boxes_coords[i, 1, 0], boxes_coords[i, 1, 1], z - half_z))
        objf.write("v {:.3f} {:.3f} {:.3f} \n".format(
            boxes_coords[i, 2, 0], boxes_coords[i, 2, 1], z - half_z))
        objf.write("v {:.3f} {:.3f} {:.3f} \n".format(
            boxes_coords[i, 3, 0], boxes_coords[i, 3, 1], z - half_z))

    # ru - rl - ll - luY
    for i in range(num_boxes):

        bi = i * 8 + 1  # !! vertex idx should begin from 1 in an obj
        objf.write("o box {} \n".format(i))
        # top: rut -> lut -> llt -> rlt
        objf.write("f {} {} {} {} \n".format(bi, bi+3, bi+2, bi+1))
        # bottom: rlb -> llb -> lub -> rub
        objf.write("f {} {} {} {} \n".format(bi+1+4, bi+2+4, bi+3+4, bi+4))
        # left: llt -> lut -> lub -> llb
        objf.write("f {} {} {} {} \n".format(bi+2, bi+3, bi+3+4, bi+2+4))
        # right: rlb -> rub -> rut -> rlt
        objf.write("f {} {} {} {} \n".format(bi+1+4, bi+4, bi, bi+1))
        # front: rlt -> llt -> llb -> rlb
        objf.write("f {} {} {} {} \n".format(bi+1, bi+2, bi+2+4, bi+1+4))
        # back: rub -> lub -> lut -> rut
        objf.write("f {} {} {} {} \n".format(bi+4, bi+3+4, bi+3, bi))

    objf.close()


def create_vol_mesh_from_params(volumes, output_mesh_path, names=None):
	num_vol = volumes.shape[0]
	# parameters of a volume
	# 0 x, 1 y, 2 z, 3 width, 4 height (of front/back face), 5 radius, 6 length of the horizontal normals (default normals point to the (0,1,0))
	# 0 FLT, 1 BLT, 2 FRT, 3 BRT, 4 FRB, 5 BRB, 6 FLB, 7 BLB
	vol_coords = np.zeros((num_vol, 8, 3))
	NV = np.sqrt(1 - volumes[:, -1] ** 2)

	Z_TOP = volumes[:, 2] + volumes[:, 5] * \
	    NV + volumes[:, 4] * volumes[:, -1] / 2
	Z_BTM = volumes[:, 2] + volumes[:, 5] * \
	    NV - volumes[:, 4] * volumes[:, -1] / 2

	Y_FRONT_TOP = volumes[:, 1] + volumes[:, 5] * \
	    volumes[:, -1] - volumes[:, 4] / 2 * NV
	Y_FRONT_BTM = volumes[:, 1] + volumes[:, 5] * \
	    volumes[:, -1] + volumes[:, 4] / 2 * NV
	Y_BACK_TOP = volumes[:, 1] - volumes[:, 5] * \
	    volumes[:, -1] + volumes[:, 4] / 2 * NV
	Y_BACK_BTM = volumes[:, 1] - volumes[:, 5] * \
	    volumes[:, -1] - volumes[:, 4] / 2 * NV

	X_RIGHT = volumes[:, 0] + volumes[:, 3] / 2
	X_LEFT = volumes[:, 0] - volumes[:, 3] / 2

	vol_coords[:, 0, :] = np.vstack(
	    [X_LEFT, Y_FRONT_TOP, Z_TOP]).transpose()  # FLT
	vol_coords[:, 1, :] = np.vstack(
	    [X_RIGHT, Y_FRONT_TOP, Z_TOP]).transpose()  # FRT
	vol_coords[:, 2, :] = np.vstack(
	    [X_LEFT, Y_FRONT_BTM, Z_BTM]).transpose()  # FLB
	vol_coords[:, 3, :] = np.vstack(
	    [X_RIGHT, Y_FRONT_BTM, Z_BTM]).transpose()  # FRB

	vol_coords[:, 4, :] = np.vstack(
	    [X_LEFT, Y_BACK_TOP, Z_TOP]).transpose()  # BLT
	vol_coords[:, 5, :] = np.vstack(
	    [X_RIGHT, Y_BACK_TOP, Z_TOP]).transpose()  # BRT
	vol_coords[:, 6, :] = np.vstack(
	    [X_LEFT, Y_BACK_BTM, Z_BTM]).transpose()  # BLB
	vol_coords[:, 7, :] = np.vstack(
	    [X_RIGHT, Y_BACK_BTM, Z_BTM]).transpose()  # BRB

	objf = open(output_mesh_path, 'w')
	objf.write('Ka 1.000000 1.000000 1.000000\n')
	objf.write('Kd 1.000000 1.000000 1.000000\n')
	objf.write('Ks 0.000000 0.000000 0.000000\n')
	objf.write('Tr 1.000000\n')
	objf.write('illum 1\n')
	objf.write('Ns 0.000000 1\n')

	# write vertices
	for i in range(num_vol):
		for j in range(8):
			objf.write("v {:.3f} {:.3f} {:.3f} \n".format(
				vol_coords[i, j, 0],
				vol_coords[i, j, 1],
				vol_coords[i, j, 2]))
	
	# write faces
	if names is None:
		names = [str(i+1) for i in range(num_vol)]

	for i in range(num_vol):
		bi = i * 8 + 1  # !! vertex idx should begin from 1 in an obj
		objf.write("o solid {} \n".format(names[i]))
        # 0 FLT, 1 FRT, 2 FLB, 3 FRB, 4 BLT, 5 BRT, 6 BLB, 7 BRB

        # top: FLT -> BLT -> BRT -> FRT 
		objf.write("f {} {} {} {} \n".format(bi, bi+4, bi+5, bi+1))
        # bottom: FLB ->FRB -> BRB -> BLB
		objf.write("f {} {} {} {} \n".format(bi+2, bi+3, bi+7, bi+6))
        # front: FLT -> FRT -> FRB -> FLB
		objf.write("f {} {} {} {} \n".format(bi, bi+1, bi+3, bi+2))
        # back: BLT -> BLB -> BRB -> BRT
		objf.write("f {} {} {} {} \n".format(bi+4, bi+6, bi+7, bi+5))
        # left: FLT -> FLB -> BLB -> BLT
		objf.write("f {} {} {} {} \n".format(bi, bi+2, bi+6, bi+4))
        # right: FRT -> BRT -> BRB -> FRB
		objf.write("f {} {} {} {} \n".format(bi+1, bi+5, bi+7, bi+3))

	objf.close()


def get_frame_of_a_volume(volume):
	wireframe = o3d.geometry.LineSet()
	points = np.zeros((8,3))
	x, y, z, width, height, r, cos = volume
	sin = np.sqrt(1 - cos ** 2)
	z_top = z + r * sin + height * cos / 2
	z_btm = z + r * sin - height * cos / 2
	y_front_top = y + r * cos - height / 2 * sin
	y_front_btm = y + r * cos + height / 2 * sin
	y_back_top = y - r * cos + height / 2 * sin
	y_back_btm = y - r * cos - height / 2 * sin
	x_right = x + width / 2
	x_left = x - width / 2
	
	points[0, :] = x_left, y_front_top, z_top # FLT
	points[1, :] = x_right, y_front_top, z_top # FRT
	points[2, :] = x_left, y_front_btm, z_btm # FLB
	points[3, :] = x_right, y_front_btm, z_btm # FRB
	points[4, :] = x_left, y_back_top, z_top # BLT
	points[5, :] = x_right, y_back_top, z_top # BRT
	points[6, :] = x_left, y_back_btm, z_btm # BLB
	points[7, :] = x_right, y_back_btm, z_btm # BRB

	lines = np.zeros((12, 2)).astype(int)
    # front 
	lines[0, :] = 0, 1 # FLT - FRT
	lines[1, :] = 2, 3 # FLB - FRB
	lines[2, :] = 0, 2 # FLT - FLB
	lines[3, :] = 1, 3 # FRT - FRB
	# back
	lines[4, :] = 4, 5 # BLT - BRT
	lines[5, :] = 6, 7 # BLB - BRB
	lines[6, :] = 4, 6 # BLT - BLB
	lines[7, :] = 5, 7 # BRT - BRB
	# top
	lines[8, :] = 0, 4 # FLT - BLT
	lines[9, :] = 1, 5 # FRT - BRT
    # bottom
	lines[10, :] = 2, 6 # FLB - BLB
	lines[11, :] = 3, 7 # FRB - BRB
    # left & right: encoding in the other four faces
	
	colors = [[0,0,0] for i in range(len(lines))]
	wireframe.points = o3d.utility.Vector3dVector(points)
	wireframe.lines = o3d.utility.Vector2iVector(lines)
	wireframe.colors = o3d.utility.Vector3dVector(colors)
	return wireframe