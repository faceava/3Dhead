import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
import matplotlib.pyplot as plt
from math import cos, sin
import skimage, pickle
from skimage import color
import os
from flask import Flask, request, send_file
from flask_cors import CORS
import uuid

# 初始化Flask应用
app = Flask(__name__)
# 启用CORS（跨域资源共享），允许前端跨域请求
CORS(app)
# 获取项目根目录绝对路径
project_root = os.path.dirname(os.path.abspath(__file__))

def look_img(img):
    img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img_RGB)
    plt.show()

def cal_eyes(LOeye, ROeye, LIeye, RIeye):
    LCeye = LOeye + (LIeye - LOeye) / 2
    RCeye = RIeye + (ROeye - RIeye) / 2
    dis_eye = np.linalg.norm(LCeye - RCeye)
    return dis_eye

def cal_angle(LOeye, ROeye, LIeye, RIeye, Nose):
    angle = np.zeros(3)
    LCeye = LOeye + (LIeye - LOeye) / 2
    RCeye = RIeye + (ROeye - RIeye) / 2
    Ceye = LCeye + (RCeye - LCeye) / 2
    angle[0] = np.arctan((Ceye[1] - Nose[1]) / (Ceye[2] - Nose[2]))
    angle[1] = np.arctan((LCeye[2] - Ceye[2]) / (LCeye[0] - Ceye[0]))/2 + np.arctan((RCeye[2] - Ceye[2]) / (RCeye[0] - Ceye[0]))/2
    angle[2] = np.arctan((Ceye[0] - Nose[0]) / (Ceye[1] - Nose[1]))
    return angle

def equal(points):
    right_face = (109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 108, 69, 104, 68, 71, 139, 34, 227, 137, 177, 215, 138, 135, 169, 170, 140, 171, 107, 66, 105, 63, 70, 156, 143, 116, 123, 147, 213, 192, 214, 210, 211, 32, 208, 55, 65, 52, 53, 46, 124, 35, 111, 117, 118, 119, 120, 121, 128, 245, 193, 221, 222, 223, 224, 225, 113, 226, 31, 228, 229, 230, 231, 232, 233, 244, 189, 56, 28, 27, 29, 30, 247, 130, 25, 110, 24, 23, 22, 26, 112, 243, 190, 157, 158, 159, 160, 161, 246, 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 122, 188, 114, 47, 100, 142, 129, 98, 196, 174, 217, 126, 209, 3, 236, 198, 49, 102, 51, 134, 131, 48, 64, 235, 240, 45, 220, 115, 219, 59, 75, 44, 237, 218, 166, 60, 99, 97, 125, 241, 238, 239, 79, 20, 242, 141, 101, 36, 203, 50, 205, 206, 187, 207, 216, 212, 202, 204, 194, 201, 167, 165, 92, 186, 57, 43, 106, 182, 83, 61, 76, 62, 78, 185, 184, 183, 191, 95, 96, 77, 146, 40, 74, 42, 80, 88, 89, 90, 91, 39, 73, 41, 81, 178, 179, 180, 181, 37, 72, 38, 82, 87, 86, 85, 84)
    left_face = (338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 337, 299, 333, 298, 301, 368, 264, 447, 366, 401, 435, 367, 364, 394, 395, 369, 396, 336, 296, 334, 293, 300, 383, 372, 345, 352, 376, 433, 416, 434, 430, 431, 262, 428, 285, 295, 282, 283, 276, 353, 265, 340, 346, 347, 348, 349, 350, 357, 465, 417, 441, 442, 443, 444, 445, 342, 446, 261, 448, 449, 450, 451, 452, 453, 464, 413, 286, 258, 257, 259, 260, 467, 359, 255, 339, 254, 253, 252, 256, 341, 463, 414, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 351, 412, 343, 277, 329, 371, 358, 327, 419, 399, 437, 355, 429, 248, 456, 420, 279, 331, 281, 363, 360, 278, 294, 455, 460, 275, 440, 344, 439, 289, 305, 274, 457, 438, 392, 290, 328, 326, 354, 461, 458, 459, 309, 250, 462, 370, 330, 266, 423, 280, 425, 426, 411, 427, 436, 432, 422, 424, 418, 421, 393, 391, 322, 410, 287, 273, 335, 406, 313, 291, 306, 292, 308, 409, 408, 407, 415, 324, 325, 307, 375, 270, 304, 272, 310, 318, 319, 320, 321, 269, 303, 271, 311, 402, 403, 404, 405, 267, 302, 268, 312, 317, 316, 315, 314)
    for i in range(0, len(left_face)):
        points[left_face[i]] = points[right_face[i]]
        points[left_face[i]][0] = points[2][0]*2 - points[right_face[i]][0]
    return points

def read_obj_skin(filename, partnum):
    read_mode = 0
    line_num = 0
    start_line = 1
    mesh_points = []
    vt_points = []
    faces1 = []
    faces2 = []
    for line in open(filename):
        values = line.split()
        if not values:
            line_num += 1
            continue
        if values[0] == 'g' and values[1] == 'default':
            read_mode += 1
        if values[0] == 'v' and read_mode == partnum:
            if start_line == 1:
                start_line_num = line_num
                start_line = 0
            coordinate = [float(values[1]), float(values[2]), float(values[3])]
            mesh_points.append(coordinate)
        if values[0] == 'vt' and read_mode == partnum:
            vt_coordinate = [float(values[1]), float(values[2])]
            vt_points.append(vt_coordinate)
        if values[0] == 'f' and read_mode == partnum:
            face = [int(values[1][0:values[1].find('/')]), int(values[2][0:values[2].find('/')]), int(values[3][0:values[3].find('/')])]
            faces1.append(face)
            face = [int(values[1][values[1].find('/')+1:values[1].rfind('/')]), int(values[2][values[2].find('/')+1:values[2].rfind('/')]),
                    int(values[3][values[3].find('/')+1:values[3].rfind('/')])]
            faces2.append(face)
        line_num += 1
    mesh_points = np.array(mesh_points)
    vt_points = np.array(vt_points)
    faces1 = np.array(faces1)
    faces2 = np.array(faces2)
    return mesh_points, vt_points, faces1, faces2, start_line_num

def angle2matrix(angles):
    x, y, z = angles[0], angles[1], angles[2]
    Rx = np.array([[1, 0, 0],
                   [0, cos(x), -sin(x)],
                   [0, sin(x), cos(x)]])
    Ry = np.array([[cos(y), 0, sin(y)],
                   [0, 1, 0],
                   [-sin(y), 0, cos(y)]])
    Rz = np.array([[cos(z), -sin(z), 0],
                   [sin(z), cos(z), 0],
                   [0, 0, 1]])
    R = Rz.dot(Ry.dot(Rx))
    return R.astype(np.float32)

def match_color_in_yuv(src_tex, dst_tex, mask):
    dst_tex_yuv = color.convert_colorspace(dst_tex, "rgb", "yuv")
    src_tex_yuv = color.convert_colorspace(src_tex, "rgb", "yuv")
    is_valid_src = mask[:, :, 0] > 0.5
    is_valid_dst = dst_tex_yuv[:, :, :] > -100
    mu_dst = np.mean(dst_tex_yuv[is_valid_dst], axis=0, keepdims=True)
    std_dst = np.std(dst_tex_yuv[is_valid_dst], axis=0, keepdims=True)
    mu_src = np.mean(src_tex_yuv[is_valid_src], axis=0, keepdims=True)
    std_src = np.std(src_tex_yuv[is_valid_src], axis=0, keepdims=True)
    match_tex_yuv = (src_tex_yuv - mu_src) / std_src
    match_tex_yuv = (match_tex_yuv / 1.1) * std_dst + mu_dst
    match_tex = skimage.color.convert_colorspace(match_tex_yuv, "yuv", "rgb")
    match_tex = np.clip(match_tex, 0, 1)
    return match_tex

def mean_in_yuv(triangle_index, points_x, points_y, img):
    tr_pt1 = np.array([points_x[triangle_index[0]], -points_y[triangle_index[0]]], np.int32)
    tr_pt2 = np.array([points_x[triangle_index[1]], -points_y[triangle_index[1]]], np.int32)
    tr_pt3 = np.array([points_x[triangle_index[2]], -points_y[triangle_index[2]]], np.int32)
    triangle = np.array([tr_pt1, tr_pt2, tr_pt3])
    rect = cv2.boundingRect(triangle)
    (x, y, w, h) = rect
    cropped_triangle = img[y: y + h, x: x + w]
    tex_yuv = color.convert_colorspace(cropped_triangle, "rgb", "yuv")
    is_valid = tex_yuv[:, :, :] > -100
    mean_yuv = np.mean(tex_yuv[is_valid], axis=0, keepdims=True)
    return mean_yuv

def transparent_back(img1, img2):
    img2 = cv2.bitwise_not(img2)
    image = cv2.add(img1, img2)
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = image.convert('RGBA')
    L, H = image.size
    color_0 = (255, 255, 255, 255)
    for h in range(H):
        for l in range(L):
            dot = (l, h)
            color_1 = image.getpixel(dot)
            if color_1 == color_0:
                color_1 = color_1[:-1] + (0,)
                image.putpixel(dot, color_1)
    return image

def process_image(input_image_path):
    """
    处理输入图片，生成3D人脸模型OBJ文件

    参数:
        input_image_path (str): 输入图片的文件路径

    返回:
        str: 生成的OBJ文件路径

    处理流程:
    1. 使用MediaPipe检测人脸关键点
    2. 计算人脸特征点的三维坐标
    3. 调整标准3D模型以匹配输入人脸
    4. 生成并保存新的OBJ文件
    """
    mp_face_mesh = mp.solutions.face_mesh
    model = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=5,
        min_tracking_confidence=0.5,
        min_detection_confidence=0.5,
    )
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=[255,230,50])

    img = cv2.imread(input_image_path)
    if img is None:
        raise Exception("无法读取输入图片")
    h,w = img.shape[0],img.shape[1]
    temp_img = img.copy()
    results = model.process(img)

    if not results.multi_face_landmarks:
        raise Exception("未检测出人脸")

    cv2.imwrite(os.path.join(project_root, 'feature.jpg'), temp_img)

    coords = np.array(results.multi_face_landmarks[0].landmark)

    def get_x(each): return each.x
    def get_y(each): return each.y
    def get_z(each): return each.z

    points_x = np.array(list(map(get_x,coords))) * w
    delx = max(points_x) - min(points_x)
    points_y = np.array(list(map(get_y,coords))) * -h
    dely = max(points_y) - min(points_y)
    points_z = np.array(list(map(get_z,coords)))
    delz = max(points_z) - min(points_z)
    points_z = points_z * delx / delz * -0.652

    points = np.vstack((points_x,points_y,points_z)).T

    User_cal_eye = cal_eyes(points[33], points[263], points[133], points[362])
    Standard_cal_eye = 69.07120

    points = points * Standard_cal_eye / User_cal_eye
    points = points + (np.zeros(3) - points[2])

    User_cal_angle = cal_angle(points[33], points[263], points[133], points[362], points[2])
    Standard_cal_angle = [-1.1419, 0, 0]

    R = angle2matrix(User_cal_angle - Standard_cal_angle)
    points = points.dot(R.T)
    points = points + ([0, -3.438599, 15.803604] - points[2])

    del_Reye = points[473] - [35.397, 47.968, -6.769]
    del_Leye = points[468] - [-35.397, 47.968, -6.769]
    del_mouth = ((points[291] - [27.824, -24.656, -4.314]) + (points[61] - [-27.829, -24.650, -4.314]))/2
    points = points[0:468]

    img2 = cv2.imread(os.path.join(project_root, "canonical_face.png"))
    if img2 is None:
        raise Exception("找不到canonical_face.png文件")
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    height, width, channels = img2.shape
    img2_new_face = np.zeros((height, width, channels), np.uint8)

    pickle_path = os.path.join(project_root, 'mediafaceS.pickle')
    if not os.path.exists(pickle_path):
        raise Exception("找不到mediafaceS.pickle文件")
    with open(pickle_path, 'rb') as file:
        standard_model = pickle.load(file)

    points2 = standard_model[1]
    points2[:,0] = points2[:, 0] * width
    points2[:,1] = points2[:, 1] * height
    points2 = np.array(points2, np.int32)
    triangles1 = standard_model[2] - 1
    triangles2 = standard_model[3] - 1

    convexhull2 = cv2.convexHull(points2)

    for triangle_index1, triangle_index2 in zip(triangles1, triangles2):
        tr1_pt1 = np.array([points_x[triangle_index1[0]], -points_y[triangle_index1[0]]], np.int32)
        tr1_pt2 = np.array([points_x[triangle_index1[1]], -points_y[triangle_index1[1]]], np.int32)
        tr1_pt3 = np.array([points_x[triangle_index1[2]], -points_y[triangle_index1[2]]], np.int32)
        triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3])

        rect1 = cv2.boundingRect(triangle1)
        (x, y, w, h) = rect1
        cropped_triangle = img[y: y + h, x: x + w]

        tri_points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                           [tr1_pt2[0] - x, tr1_pt2[1] - y],
                           [tr1_pt3[0] - x, tr1_pt3[1] - y]])

        tr2_pt1 = [points2[triangle_index2[0]][0], points2[triangle_index2[0]][1]]
        tr2_pt2 = [points2[triangle_index2[1]][0], points2[triangle_index2[1]][1]]
        tr2_pt3 = [points2[triangle_index2[2]][0], points2[triangle_index2[2]][1]]
        triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3])

        rect2 = cv2.boundingRect(triangle2)
        (x, y, w, h) = rect2

        cropped_tr2_mask = np.zeros((h, w), np.uint8)

        tri_points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                            [tr2_pt2[0] - x, tr2_pt2[1] - y],
                            [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

        cv2.fillConvexPoly(cropped_tr2_mask, tri_points2, 255)

        tri_points = np.float32(tri_points)
        tri_points2 = np.float32(tri_points2)
        M = cv2.getAffineTransform(tri_points, tri_points2)
        warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

        img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
        img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
        _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

        img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
        img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area

    img2_face_mask = np.zeros_like(img2_gray)
    temp_img = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
    img2_face_mask = cv2.bitwise_not(temp_img)

    temp_img = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
    result_img = cv2.add(temp_img, img2_new_face)

    cv2.flip(result_img, 0, img2)

    cv2.imwrite(os.path.join(project_root, 'facegenS_skin.jpg'), img2)

    img2small = cv2.resize(img2, (346, 346))
    masksmall_path = os.path.join(project_root, "FaceMaskSmall.png")
    if not os.path.exists(masksmall_path):
        raise Exception("找不到FaceMaskSmall.png文件")
    masksmall = cv2.imread(masksmall_path)
    temp_img = transparent_back(img2small, masksmall)

    img1_path = os.path.join(project_root, "face_skin_whole.png")
    if not os.path.exists(img1_path):
        raise Exception("找不到face_skin_whole.png文件")
    img1 = Image.open(img1_path)
    img1 = img1.convert("RGBA")
    img1.paste(temp_img, (339, 622), mask=None)

    uv_texture2_path = os.path.join(project_root, "head_skinW_s.png")
    if not os.path.exists(uv_texture2_path):
        raise Exception("找不到head_skinW_s.png文件")
    uv_texture2 = Image.open(uv_texture2_path).convert("RGB")
    uv_texture_array2 = np.array(uv_texture2)

    uv_texture3_path = os.path.join(project_root, "maskskin1.jpg")
    if not os.path.exists(uv_texture3_path):
        raise Exception("找不到maskskin1.jpg文件")
    uv_texture3 = cv2.imread(uv_texture3_path)
    uv_texture3 = np.stack((uv_texture3,) * 3, axis=-1)
    uv_texture_array3 = np.array(uv_texture3, dtype=np.uint8)

    uv_texture1 = Image.fromarray(cv2.cvtColor(img2small, cv2.COLOR_BGR2RGB))
    uv_texture_array1 = np.array(uv_texture1)/255

    temp_img = match_color_in_yuv(src_tex=uv_texture_array2, dst_tex=uv_texture_array1, mask=uv_texture_array3) * 255
    dst = temp_img[:,:,[2,1,0]]
    dst = np.array(dst, dtype=np.uint8)

    src_mask_path = os.path.join(project_root, "maskskin2.jpg")
    if not os.path.exists(src_mask_path):
        raise Exception("找不到maskskin2.jpg文件")
    src_mask = cv2.imread(src_mask_path, 0)
    src_mask = cv2.merge((src_mask, src_mask, src_mask))
    center = (512, 805)

    img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
    img1 = img1.astype(np.float32) / 255.0 * src_mask
    gray_mask = cv2.cvtColor(src_mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_mask, 254, 255, cv2.THRESH_BINARY)
    mixed_clone = cv2.seamlessClone(img1, dst, mask, center, cv2.NORMAL_CLONE)

    temp_img_path = os.path.join(project_root, "head_skin_mask.png")
    if not os.path.exists(temp_img_path):
        raise Exception("找不到head_skin_mask.png文件")
    temp_img = cv2.imread(temp_img_path)
    icon2 = transparent_back(mixed_clone, temp_img)

    img2_path = os.path.join(project_root, "Models/standard_wholebodyW_s.png")
    if not os.path.exists(img2_path):
        raise Exception("找不到Models/standard_wholebodyW_s.png文件")
    img2 = Image.open(img2_path)
    temp_img = Image.new('RGBA', img2.size, (0, 0, 0, 0))
    temp_img.paste(icon2, (0, 0))
    result = Image.alpha_composite(img2.convert('RGBA'), temp_img)
    result.save(os.path.join(project_root, "Models/standard_wholebodyW_u.png"))

    obj_input_path = os.path.join(project_root, 'Models/sims_standardW_s.obj')
    if not os.path.exists(obj_input_path):
        raise Exception("找不到Models/sims_standardW_s.obj文件")
    with open(obj_input_path, 'r') as file:
        lines = file.readlines()

    new_lines = []
    count = 0
    partnum = 0
    for line in lines:
        values = line.split()
        if not values:
            continue

        if values[0] == 'g' and values[1] == 'default':
            partnum += 1
        if values[0] == 'v':
            if partnum==1 and count < 468:
                new_vertex_coords = f"{points[count,0]} {points[count,1]} {points[count,2]}"
                new_line = 'v ' + new_vertex_coords + '\n'
                new_lines.append(new_line)
                count += 1
            elif partnum==2 or partnum==3:
                new_vertex_coords = f"{float(values[1]) + del_Reye[0]} {float(values[2]) + del_Reye[1]} {float(values[3]) + min(0, del_Reye[2]*1.5)}"
                new_line = 'v ' + new_vertex_coords + '\n'
                new_lines.append(new_line)
            elif partnum == 4 or partnum == 5:
                new_vertex_coords = f"{float(values[1]) + del_Leye[0]} {float(values[2]) + del_Leye[1]} {float(values[3]) + min(0, del_Leye[2]*1.5)}"
                new_line = 'v ' + new_vertex_coords + '\n'
                new_lines.append(new_line)
            elif partnum < 11 and partnum > 5:
                new_vertex_coords = f"{float(values[1])} {float(values[2]) + del_mouth[1]} {float(values[3]) + min(0, del_mouth[2]*1.5)}"
                new_line = 'v ' + new_vertex_coords + '\n'
                new_lines.append(new_line)
            else:
                new_lines.append(line)
        elif values[0] == 'mtllib':
            new_lines.append('mtllib sims_standardW_u.mtl' + '\n')
        else:
            new_lines.append(line)

    obj_output_path = os.path.join(project_root, 'Models/sims_standardW_u.obj')
    with open(obj_output_path, 'w') as file:
        file.writelines(new_lines)
    return obj_output_path
    
"""
/upload接口 - 处理图片上传并生成3D模型OBJ文件
请求方法: POST
请求参数: 
  - file: 上传的图片文件（multipart/form-data格式）

返回值:
  - 成功: OBJ文件下载（application/octet-stream）
  - 失败: 错误信息和状态码（400或500）

处理流程:
1. 验证请求中是否包含文件
2. 生成唯一ID并保存临时文件
3. 调用process_image处理图片并生成OBJ
4. 返回OBJ文件给客户端
5. 清理临时文件
"""
@app.route('/upload', methods=['POST'])
def upload_image():
    print('收到/upload请求')
    # 检查请求中是否包含文件部分
    if 'file' not in request.files:
        print('No file part in request.files')
        return 'No file part', 400
    file = request.files['file']
    # 检查文件名是否为空
    if file.filename == '':
        print('No selected file')
        return 'No selected file', 400
    if file:
        # 生成唯一ID作为临时文件名
        unique_id = str(uuid.uuid4())
        input_path = os.path.join(project_root, f'temp_{unique_id}.jpg')
        # 保存上传的文件到临时路径
        file.save(input_path)
        print(f'文件已保存到: {input_path}')
        try:
            # 调用图片处理函数生成OBJ文件
            obj_path = process_image(input_path)
            print(f'处理完成，返回OBJ: {obj_path}')
            # 返回OBJ文件作为附件下载
            return send_file(obj_path, as_attachment=True, mimetype='application/octet-stream')
        except Exception as e:
            # 捕获处理过程中的异常并返回错误信息
            print(f'处理异常: {e}')
            return str(e), 500
        finally:
            # 确保临时文件被删除，清理资源
            if os.path.exists(input_path):
                os.remove(input_path)
                print(f'已删除临时文件: {input_path}')

if __name__ == '__main__':
    # 启动Flask服务器
    # 主机设置为0.0.0.0允许外部网络访问
    # 端口: 5000
    # debug=True: 开启调试模式，代码修改后自动重启
    app.run(host='0.0.0.0', port=5000, debug=True)