from typing import List

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
import modern_robotics as mr
try:
    from ik_geo import Robot as IKGeoRobot
except Exception:
    IKGeoRobot = None

from arm.geometry import Geometry3D, Capsule
from arm.utils import MathUtils
from .robot_config import RobotConfig
from .robot import Robot, get_transformation_mdh, wrap


class UR5e(Robot):

    def __init__(self) -> None:
        """
        Инициализирует геометрические параметры, динамические параметры и кинематическую модель манипулятора UR5e.
        """
        super().__init__()
        
        # Определяем геометрические параметры манипулятора
        d1 = 0.163  # Смещение от первого сустава до второго
        d4 = 0.134  # Смещение от четвёртого сустава до пятого
        d5 = 0.1    # Смещение от пятого сустава до шестого
        d6 = 0.1    # Смещение на конце шестого сустава

        a3 = 0.425  # Длина третьего звена
        a4 = 0.392  # Длина четвёртого звена

        self._dof = 6  # Число степеней свободы (у UR5e 6 суставов)
        self.q0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Начальные углы суставов

        # Параметры Денавита — Хартенберга (DH)
        alpha_array = [0.0, -np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2]  # Угол кручения между осями суставов
        a_array = [0.0, 0.0, a3, a4, 0.0, 0.0]  # Расстояние между осями суставов
        d_array = [d1, 0.0, 0.0, d4, d5, d6]   # Смещение между осями суставов
        theta_array = [0.0, -np.pi / 2, 0.0, np.pi / 2, 0.0, 0.0]  # Начальные углы суставов
        sigma_array = [0, 0, 0, 0, 0, 0]  # Тип сустава (0 — вращательный)

        # Определяем массу, положение центра масс, тензор инерции и инерцию мотора для каждого сустава
        m1 = 3.7  # Масса первого сустава
        r1 = np.array([0.0, 0.0, 0.0])  # Положение центра масс первого сустава
        I1 = np.diag([0.0102675, 0.0102675, 0.00666])  # Тензор инерции первого сустава
        Jm1 = 0.1  # Инерция мотора первого сустава

        m2 = 8.393  # Масса второго сустава
        r2 = np.array([0.2125, 0.0, 0.138])  # Положение центра масс второго сустава
        I2 = np.diag([0.0151074, 0.133886, 0.133886])  # Тензор инерции второго сустава
        Jm2 = 0.1  # Инерция мотора второго сустава

        m3 = 2.275  # Масса третьего сустава
        r3 = np.array([0.196, 0.0, 0.007])  # Положение центра масс третьего сустава
        I3 = np.diag([0.004095, 0.0311796, 0.0311796])  # Тензор инерции третьего сустава
        Jm3 = 0.1  # Инерция мотора третьего сустава

        m4 = 1.219  # Масса четвёртого сустава
        r4 = np.array([0.0, 0.0, 0.0])  # Положение центра масс четвёртого сустава
        I4 = np.diag([0.0025599, 0.0021942, 0.0025599])  # Тензор инерции четвёртого сустава
        Jm4 = 0.1  # Инерция мотора четвёртого сустава

        m5 = 1.219  # Масса пятого сустава
        r5 = np.array([0.0, 0.0, 0.0])  # Положение центра масс пятого сустава
        I5 = np.diag([0.0025599, 0.0025599, 0.0021942])  # Тензор инерции пятого сустава
        Jm5 = 0.1  # Инерция мотора пятого сустава

        m6 = 0.1889  # Масса шестого сустава
        r6 = np.array([0.0, 0.0, -0.0228317])  # Положение центра масс шестого сустава
        I6 = np.diag([9.90863e-05, 9.90863e-05, 0.000132134])  # Тензор инерции шестого сустава
        Jm6 = 0.1  # Инерция мотора шестого сустава

        ms = [m1, m2, m3, m4, m5, m6]  # Список масс всех суставов
        rs = [r1, r2, r3, r4, r5, r6]  # Список положений центров масс всех суставов
        Is = [I1, I2, I3, I4, I5, I6]  # Список тензоров инерции всех суставов
        Jms = [Jm1, Jm2, Jm3, Jm4, Jm5, Jm6]  # Список инерций моторов всех суставов

        links = []  # Сохраняем объекты звеньев всех суставов
        for i in range(6):  # Проходим по каждому суставу
            links.append(rtb.DHLink(d=d_array[i], alpha=alpha_array[i], a=a_array[i], offset=theta_array[i], mdh=True,
                                    m=ms[i], r=rs[i], I=Is[i], Jm=Jms[i], G=1.0))
        self.robot = rtb.DHRobot(links)  # Строим модель манипулятора

        self.alpha_array = alpha_array  # Сохраняем массив alpha
        self.a_array = a_array          # Сохраняем массив a
        self.d_array = d_array          # Сохраняем массив d
        self.theta_array = theta_array  # Сохраняем массив theta
        self.sigma_array = sigma_array  # Сохраняем массив sigma

        T = SE3()  # Инициализируем однородные матрицы преобразования
        for i in range(self.dof):  # Проходим по каждому суставу
            Ti = get_transformation_mdh(self.alpha_array[i], self.a_array[i], self.d_array[i], self.theta_array[i],
                                        self.sigma_array[i], 0.0)  # Вычисляем матрицу преобразования для каждого сустава
            self._Ms.append(Ti.A)  # Сохраняем матрицу преобразования
            T: SE3 = T * Ti  # Накапливаем матрицу преобразования
            self._Ses.append(np.hstack((T.a, np.cross(T.t, T.a))))  # Сохраняем вектор пространства

            Gm = np.zeros((6, 6))  # Инициализируем матрицу инерции
            Gm[:3, :3] = Is[i]  # Задаём часть с тензором инерции
            Gm[3:, 3:] = ms[i] * np.eye(3)  # Задаём часть с массой
            AdT = mr.Adjoint(mr.RpToTrans(np.eye(3), -rs[i]))  # Вычисляем присоединённую матрицу
            self._Gs.append(AdT.T @ Gm @ AdT)  # Сохраняем матрицу инерции
            self._Jms.append(Jms[i])  # Сохраняем инерцию мотора

        self._Ms.append(np.eye(4))  # Добавляем единичную матрицу в конце

        self.robot_config = RobotConfig()  # Инициализируем объект конфигурации робота
        self.robot_config.wrist = -1  # Задаём конфигурацию запястья по умолчанию
        self._ikgeo_robot = self._build_ikgeo_solver()

    def _build_ikgeo_solver(self):
        if IKGeoRobot is None:
            return None

        # Project-specific POE chain aligned to the MuJoCo/MDH model.
        h = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float64)
        p = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.163],
            [0.0, 0.0, 0.425],
            [0.0, 0.0, 0.392],
            [0.0, 0.134, 0.0],
            [0.0, 0.0, 0.1],
            [0.0, 0.1, 0.0],
        ], dtype=np.float64)
        try:
            return IKGeoRobot.three_parallel_two_intersecting(h, p)
        except Exception:
            return None

    def _ikgeo_frame_transforms(self):
        # Fitted once against this repository's UR5e MDH model.
        A = np.array([
            [-0.472923, -0.880551, -0.031205],
            [-0.851548,  0.447677,  0.272858],
            [-0.226295,  0.155613, -0.961548],
        ], dtype=np.float64)
        B = np.array([
            [-0.472923,  0.031205, -0.880551],
            [-0.851548, -0.272857,  0.447677],
            [-0.226295,  0.961548,  0.155613],
        ], dtype=np.float64)
        return A, B

    def _ikine_ikgeo(self, Tep: SE3) -> np.ndarray:
        if self._ikgeo_robot is None:
            return np.array([])

        target = self._base.inv() * Tep * self._tool.inv()
        A, B = self._ikgeo_frame_transforms()
        R_query = A.T @ np.array(target.R, dtype=np.float64) @ B.T
        t_query = np.array(target.t, dtype=np.float64)

        try:
            solutions = self._ikgeo_robot.get_ik_sorted(R_query, t_query)
        except Exception:
            return np.array([])

        if not solutions:
            return np.array([])

        current_q = np.array(self.q0, dtype=np.float64)
        best_q = None
        best_score = None
        for q_sol, err, least_squares in solutions:
            q_sol = np.array(q_sol, dtype=np.float64)
            if not np.all(np.isfinite(q_sol)):
                continue
            diff = np.arctan2(np.sin(q_sol - current_q), np.cos(q_sol - current_q))
            score = float(np.linalg.norm(diff))
            if least_squares:
                score += 2.0
            score += float(err) * 5.0
            if best_score is None or score < best_score:
                best_score = score
                best_q = q_sol

        if best_q is None:
            return np.array([])
        return best_q

    def _ikine_numeric(self, Tep: SE3) -> np.ndarray:
        current_q = np.array(self.q0, dtype=np.float64)
        seeds = [
            current_q,
            np.zeros(self.dof, dtype=np.float64),
            np.array([0.0, -np.pi / 2, np.pi / 2, 0.0, -np.pi / 2, 0.0], dtype=np.float64),
            current_q + np.array([0.0, -0.2, 0.2, 0.0, 0.0, 0.0], dtype=np.float64),
            current_q + np.array([0.0, 0.2, -0.2, 0.0, 0.0, 0.0], dtype=np.float64),
        ]
        ikgeo_seed = self._ikine_ikgeo(Tep)
        if len(ikgeo_seed):
            seeds.insert(0, ikgeo_seed)
        masks = [[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]]

        for seed in seeds:
            for mask in masks:
                try:
                    sol = self.robot.ikine_LM(
                        Tep,
                        q0=seed,
                        ilimit=60,
                        slimit=20,
                        tol=1e-3,
                        joint_limits=False,
                        mask=mask,
                    )
                except Exception:
                    continue

                if getattr(sol, "success", False):
                    return np.array(sol.q, dtype=np.float64)

        return np.array([])

    def ikine(self, Tep: SE3) -> np.ndarray:

        T06: SE3 = self._base.inv() * Tep * self._tool.inv()

        wpc = self.cal_wpc(T06)

        thetas = [0.0 for _ in range(self.dof)]

        # solve theta1
        theta1_condition = np.power(wpc[1], 2) + np.power(wpc[0], 2) - np.power(self.d_array[3], 2)
        if theta1_condition < 0:
            return self._ikine_numeric(Tep)
        if MathUtils.near_zero(np.linalg.norm([wpc[1], wpc[0]])):
            thetas[0] = self.q0[0] - self.theta_array[0]  # overhead singularity
        else:
            # if self.robot_config.overhead == 0:
            #     thetas[0] = np.arctan2(wpc[1], wpc[0]) - np.arctan2(self.d_array[3], np.sqrt(theta1_condition))
            # elif self.robot_config.overhead == 1:
            #     thetas[0] = np.arctan2(wpc[1], wpc[0]) - np.arctan2(self.d_array[3], -np.sqrt(theta1_condition))
            # else:
            #     return np.array([])
            thetas[0] = np.arctan2(wpc[1], wpc[0]) - np.arctan2(self.d_array[3],
                                                                self.robot_config.overhead * np.sqrt(theta1_condition))

        # solve theta5
        theta5_condition = -T06.a[0] * np.sin(thetas[0]) + T06.a[1] * np.cos(thetas[0])
        if np.abs(theta5_condition) > 1:
            return self._ikine_numeric(Tep)
        # if self.robot_config.wrist == 0:
        #     thetas[4] = np.arccos(theta5_condition)
        # elif self.robot_config.wrist == 1:
        #     thetas[4] = -np.arccos(theta5_condition)
        # else:
        #     return np.array([])
        thetas[4] = self.robot_config.wrist * np.arccos(theta5_condition)

        # solve theta6
        m1 = - T06.n[0] * np.sin(thetas[0]) + T06.n[1] * np.cos(thetas[0])
        n1 = - T06.o[0] * np.sin(thetas[0]) + T06.o[1] * np.cos(thetas[0])
        if MathUtils.near_zero(np.sin(thetas[4])):
            thetas[5] = self.q0[5] - self.theta_array[5]
        else:
            thetas[5] = np.arctan2(-n1 / np.sin(thetas[4]), m1 / np.sin(thetas[4]))

        # solve theta3
        T01 = get_transformation_mdh(self.alpha_array[0], self.a_array[0], self.d_array[0], thetas[0],
                                     self.sigma_array[0],
                                     0.0)
        T45 = get_transformation_mdh(self.alpha_array[4], self.a_array[4], self.d_array[4], thetas[4],
                                     self.sigma_array[4],
                                     0.0)
        T56 = get_transformation_mdh(self.alpha_array[5], self.a_array[5], self.d_array[5], thetas[5],
                                     self.sigma_array[5],
                                     0.0)
        T46 = T45 * T56
        T14 = T01.inv() * T06 * T46.inv()
        x = T14.t[0]
        y = T14.t[2]

        theta3_condition = (np.power(x, 2) + np.power(y, 2) - np.power(self.a_array[2], 2)
                            - np.power(self.a_array[3], 2)) / (2 * self.a_array[2] * self.a_array[3])
        if np.abs(theta3_condition) > 1.0:
            return self._ikine_numeric(Tep)
        # if self.robot_config.inline == 0:
        #     thetas[2] = np.arccos(theta3_condition)
        # elif self.robot_config.inline == 1:
        #     thetas[2] = - np.arccos(theta3_condition)
        # else:
        #     return np.array([])
        thetas[2] = self.robot_config.inline * np.arccos(theta3_condition)

        # solve theta2
        M = np.array([
            [self.a_array[3] * np.cos(thetas[2]) + self.a_array[2], -self.a_array[3] * np.sin(thetas[2])],
            [self.a_array[3] * np.sin(thetas[2]), self.a_array[3] * np.cos(thetas[2]) + self.a_array[2]]
        ])
        XY = np.array([
            [x],
            [-y]
        ])
        CS = np.linalg.inv(M) @ XY
        thetas[1] = np.arctan2(CS[1, 0], CS[0, 0])

        # solve theta4
        thetas[3] = np.arctan2(-T14.o[0], T14.n[0]) - thetas[1] - thetas[2]

        qs = np.array([wrap(thetas[i] - self.theta_array[i])[0] for i in range(self.dof)])

        q0_s = list(map(wrap, self.q0))

        for i in range(self.dof):
            if qs[i] - q0_s[i][0] > np.pi:
                qs[i] += (q0_s[i][1] - 1) * 2 * np.pi
            elif qs[i] - q0_s[i][0] < -np.pi:
                qs[i] += (q0_s[i][1] + 1) * 2 * np.pi
            else:
                qs[i] += q0_s[i][1] * 2 * np.pi

        if np.all(np.isfinite(qs)):
            return qs

        return self._ikine_numeric(Tep)

    def set_robot_config(self, q):
        T = self.fkine(q)
        wpc = self.cal_wpc(T)
        thetas = [q[i] + self.theta_array[i] for i in range(self.dof)]

        if wrap(np.arctan2(wpc[1], wpc[0]) - wrap(thetas[0])[0])[0] <= np.pi / 2:
            self.robot_config.overhead = 1
        else:
            self.robot_config.overhead = -1

        # inline
        if wrap(thetas[2])[0] >= 0:
            self.robot_config.inline = 1
        else:
            self.robot_config.inline = -1

        # wrist
        if wrap(thetas[4])[0] >= 0:
            self.robot_config.wrist = 1
        else:
            self.robot_config.wrist = -1

    def cal_wpc(self, T: SE3) -> np.ndarray:
        t = T.t - self.d_array[5] * T.a
        t[2] -= self.d_array[0]
        return t

    def get_geometries(self) -> List[Geometry3D]:
        Ts = []
        T = SE3()
        for i in range(self.dof):
            T = T * get_transformation_mdh(self.alpha_array[i], self.a_array[i], self.d_array[i], self.theta_array[i],
                                           self.sigma_array[i], self.q0[i])
            Ts.append(T)

        T1 = Ts[0] * SE3.Trans(0, 0, -0.04)
        geometry1 = Capsule(T1, 0.06, 0.12)

        T2 = Ts[1] * SE3.Trans(0.2, 0, 0.138) * SE3.Ry(-np.pi / 2)
        geometry2 = Capsule(T2, 0.05, 0.4)

        T3 = Ts[2] * SE3.Trans(0.2, 0, 0.007) * SE3.Ry(-np.pi / 2)
        geometry3 = Capsule(T3, 0.038, 0.38)

        T4 = Ts[3] * SE3.Trans(0.0, 0.0, -0.07)
        geometry4 = Capsule(T4, 0.04, 0.14)

        T5 = Ts[4] * SE3.Trans(0.0, 0.0, -0.06)
        geometry5 = Capsule(T5, 0.04, 0.12)

        T6 = Ts[5] * SE3.Trans(0.0, 0.0, -0.06)
        geometry6 = Capsule(T6, 0.04, 0.12)

        return [geometry1, geometry2, geometry3, geometry4, geometry5, geometry6]


if __name__ == '__main__':
    ur_robot = UR5e()
    q0 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    T1 = ur_robot.fkine(q0)
    print(T1)
    ur_robot.move_cartesian(T1)
    q_new = ur_robot.get_joint()
    print(q_new)
