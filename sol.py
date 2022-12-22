import numpy as np
from cubelit import cublit


def map_cube(lo):
    return cublit(lo)


motor = ["t", 'B', "r", "l", 'f', 'b']
color = ['y', 'w', 'r', 'o', 'b', 'g']
sides = motor[2:]
orient = [(0, 0), (0, 2), (1, 2), (1, 1), (2, 0), (2, 2)]  # (axis, index),

ang = 90
# cube = np.zeros((3, 3, 3))
# cube = np.array([[[1, 2, 3], [10, 11, 12], [19, 20, 21]],
#                  [[4, 5, 6], [13, 14, 15], [22, 23, 24]],
#                  [[7, 8, 9], [16, 17, 18], [25, 26, 27]]])
# cube = np.array([[[1, 10, 19], [2, 11, 20], [3, 12, 21]],
#                  [[4, 13, 22], [5, 14, 23], [6, 15, 24]],
#                  [[7, 16, 25], [8, 17, 26], [9, 18, 27]]])
# 0,0,0 = front top left, 2,2,2 = back, bot, right

matrix_t = [[[0, 0, 2], (0, 1, -1)],  # top
            [[2, 0, 0], (0, 1, 1)],  # bot
            [[0, 2, 0], (1, 0, 1)],
            [[0, 0, 2], (1, 0, -1)],
            [[0, 0, 0], (1, 1, 0)],
            [[0, 2, 2], (1, -1, 0)]]

cube_col_r1 = [[('b', 'r', 'w'), ('0', 'o', 'b'), ('o', 'b', 'y')],
               [('y', '0', 'o'), ('0', '0', 'y'), ('w', '0', 'b')],
               [('g', 'o', 'y'), ('0', 'g', 'y'), ('w', 'o', 'b')]]

cube_col_r2 = [[('o', 'g', '0'), ('0', 'o', '0'), ('r', 'y', '0')],
               [('b', '0', '0'), ('0', '0', '0'), ('g', '0', '0')],
               [('b', 'y', '0'), ('0', 'r', '0'), ('w', 'g', '0')]]

cube_col_r3 = [[('y', 'r', 'g'), ('0', 'r', 'b'), ('g', 'w', 'r')],
               [('o', '0', 'w'), ('0', '0', 'w'), ('r', '0', 'g')],
               [('y', 'r', 'b'), ('0', 'w', 'r'), ('o', 'g', 'w')]]

cube_ar = [cube_col_r1, cube_col_r2, cube_col_r3]
cube = []
for r in cube_ar:
    for c in r:
        for d in c:
            cube.append(cublit(d))
            
cube = np.array(cube).reshape((3, 3, 3))
# cube = np.vectorize(map_cube)(cube_ar)


def get_locals(st, trans=False):
    old_ord = []
    le = 3
    for i in range(le):
        loc_ful = st[0][:]
        for j in range(le):
            for k in range(le):
                loc_ful[0] = st[0][0] + st[1][0] * i
                if trans:
                    kk = j
                    jj = k
                else:
                    kk = k
                    jj = j
                loc_ful[1] = st[0][1] + st[1][1] * jj
                loc_ful[2] = st[0][2] + st[1][2] * kk
                old_ord.append(tuple(loc_ful))

    old_ord = list(dict.fromkeys(old_ord))
    return old_ord


def turn_mat(axis, clock=True, cnt=1):
    mot_n = motor.index(axis)
    st = matrix_t[mot_n]
    for n in range(cnt):
        old_ord = get_locals(st)
        shuffle_ord = np.array(old_ord, "i,i,i").reshape((3, 3))  # todo order
        if clock:
            shuffle_ord = np.transpose(shuffle_ord)  # todo return?
            shuffle_ord = np.fliplr(shuffle_ord)
        else:
            shuffle_ord = np.flipud(shuffle_ord)
            shuffle_ord = np.transpose(shuffle_ord)
        old_cube = cube.copy()
        shuffle_ord = shuffle_ord.flatten()
        for o_loc, n_loc in zip(old_ord, shuffle_ord):
            n_loc = list(n_loc)
            # # o_loc = np.array(loc_o)
            # print(n_loc, o_loc, sep=', ')
            cube[n_loc[0]][n_loc[1]][n_loc[2]] = old_cube[o_loc[0]][o_loc[1]][o_loc[2]]

    # if clock ^ sum(st[1]) <= 0:
    #     d = 1
    # else:
    #     d = -1
    # rotate_cublits(shuffle_ord, st[1].index(0), d)


def rotate_cublits(cubelits, axis):
    for c2 in cubelits:
        c2.rotate(axis)


def print_full():
    print_fu = []
    for ax in motor:
        print_fu.append(print_side(ax))
    # ['y', 'w', 'r', 'o', 'b', 'g']
    n_p = [bordered(s) for s in print_fu]
    n_p2 = con_bord(n_p[3], n_p[4], n_p[2])
    n_p[2] = n_p2
    size = max(len(s.split('\n')[1]) for s in n_p)
    fp = [n_p[n] for n in [0, 2, 1, 5]]
    n_str = '{:^'+str(size)+'}'
    form = f'{n_str}\n\n{n_str}\n\n{n_str}\n\n{n_str}'
    print(form.format(*fp))
    print('*'*20)
    # print(n_p[2])
    # print(n_p[1])
    # print(n_p[5])


def print_side(axis):
    mot_n = motor.index(axis)
    st = matrix_t[mot_n]
    cube_side = get_locals(st, mot_n < 2)
    print_arr = []
    # print_arr = np.zeros(cube_side.shape[0])
    axis_xyz = 2 - mot_n // 2  # is xyz
    for ind in cube_side:
        pnt_val = cube[ind[0]][ind[1]][ind[2]].print_face(axis_xyz)  # disp all faces in that dir?
        print_arr.append(pnt_val)
    print_arr = np.array(print_arr).reshape((3, 3))
    return print_arr


def turn(mot, clock=True):
    mot_n = motor.index(mot)
    di = 1 if clock else -1
    turn_n = mot_n * di * ang  # todo mot rotate physical
    return turn_n


def sence():
    pass


def relitive_or():
    pass


def cross(col, loc):
    # find color
    centers = [[2, 1], [0, 1], [1, 0], [1, 2]]
    index = color.index(col) - 2
    goal = [2] + centers[index]
    if index % 2 == 0:  # front right
        mul = 1
    else:
        mul = -1
    if index < 2:  # right left
        col_check = 2
    else:
        col_check = 1
    tot = (loc[col_check] - goal[col_check]) * mul
    rel_index = (index + tot) % 4

    if abs(tot) == 2:
        if loc[0] != 0:
            turn_mat(motor[rel_index], cnt=2)
        turn_mat('t', cnt=2)
        turn_mat(motor[index], cnt=2)
    elif tot != 0:
        x = tot < 0  # for left
        if loc[0] == 0:
            turn_mat('t', x)
            turn_mat(motor[index], x)

        # if line 2 and opisidse check is not same
        elif loc[0] == 1 and loc[col_check % 2 + 1] - goal[col_check % 2 + 1] != 0:
            turn_mat(motor[rel_index], x, 2)
        elif loc[0] == 2:
            turn_mat(motor[rel_index], x)
        turn_mat(motor[index], x)

    w_d = cube[goal[0], goal[1], goal[2]].index('w')  # find white
    if w_d != 2 and w_d != col_check - 1:
        rel_2 = (index + 1) % 4
        turn_mat(motor[index], False)
        turn_mat(motor[rel_2])
        turn_mat('t')
        turn_mat(motor[rel_2], False)
        turn_mat(motor[index], cnt=2)


def full_cross():
    pass


def beginer_corner(col, loc):
    # todo check rel axis, orient
    # bot, top
    # -right
    # -front
    # - top
    """ """
    alg = ["rbt2b'r'",  # top top
           "trt'r'",  # top for
           "t'f'tf",  # top right
           "rt2r'f't2f",  # bot right
           "f't2frt2r'"  # bot front
           ]


def l2():
    pass


def con_bord(*mtext):
    ntext = [n.split('\n') for n in mtext]
    max_len = 5
    res = []
    for n in range(max_len):
        lines = [li[n] for li in ntext]  # splitext?
        res.append(' '.join(lines))

    return '\n'.join(res)


def bordered(mat):
    lines = [row for row in mat]  # todo needed
    width = max(len(s) for s in lines)*3-2  # -2 for last
    res = ['┌' + '─' * width + '┐']
    for s in lines:
        res.append('│' + (', '.join(s) + ' ' * width)[:width] + '│')
    res.append('└' + '─' * width + '┘')
    return '\n'.join(res)


# def f2l(col, loc1, loc2):
#
#     """
#     let x be rewlitive x and right, top be relitive
#     case: color above hole oposite sides, opisite colors
#     # on right: RTR'
#     Case: white top left of hole(front right), RT2R' TRT'R'
#     """
#     case = ['in bot', 'wt', 'next, correct', 'next, incorrect',
#             'adjasent, same', 'opisote sme', 'adjasent, dif', 'opisote dif']
#     # find color
#     # todo move so qhitte corner above
#     # white top
#
# def full_f2l():
#     pass


# testing
print_full()
turn_mat('f')
print_full()
# turn_mat('l')
# print(cube)
