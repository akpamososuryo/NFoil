import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

@ti.func
def panel_info_ti(x1, z1, x2, z2, xi, zi):
    dx = x2 - x1
    dz = z2 - z1
    d = ti.math.sqrt(dx*dx + dz*dz)
    
    t_x = dx / d
    t_z = dz / d
    n_x = -t_z
    n_z = t_x
    
    dx1 = xi - x1
    dz1 = zi - z1
    x = dx1*t_x + dz1*t_z
    z = dx1*n_x + dz1*n_z
    
    dx2 = xi - x2
    dz2 = zi - z2
    r1 = ti.math.sqrt(dx1*dx1 + dz1*dz1)
    r2 = ti.math.sqrt(dx2*dx2 + dz2*dz2)
    
    theta1 = ti.math.atan2(z, x)
    theta2 = ti.math.atan2(z, x-d)
    
    return t_x, t_z, n_x, n_z, x, z, d, r1, r2, theta1, theta2

@ti.func
def panel_vortex_velocity_ti(x1, z1, x2, z2, xi, zi):
    t_x, t_z, n_x, n_z, x, z, d, r1, r2, theta1, theta2 = panel_info_ti(x1, z1, x2, z2, xi, zi)
    
    ug1 = 0.0; ug2 = 0.0
    wg1 = 0.0; wg2 = 0.0
    
    if d >= 1e-15:
        logr1 = 0.0
        logr2 = 0.0
        if r1 >= 1e-9: 
            logr1 = ti.math.log(r1)
        else: 
            theta1 = ti.math.pi
            theta2 = ti.math.pi
            
        if r2 >= 1e-9: 
            logr2 = ti.math.log(r2)
        else: 
            theta1 = 0.0
            theta2 = 0.0
            
        pi2 = 2.0 * ti.math.pi
        
        temp1 = (theta2 - theta1) / pi2
        temp2 = (2.0*z*(logr1 - logr2) - 2.0*x*(theta2 - theta1)) / (2.0 * pi2 * d)
        ug1 = temp1 + temp2
        ug2 = -temp2
        
        temp1 = (logr2 - logr1) / pi2
        temp2 = (x*(logr1 - logr2) - d + z*(theta2 - theta1)) / (pi2 * d)
        wg1 = temp1 + temp2
        wg2 = -temp2

    a_x = ug1*t_x + wg1*n_x; a_z = ug1*t_z + wg1*n_z
    b_x = ug2*t_x + wg2*n_x; b_z = ug2*t_z + wg2*n_z
    return a_x, a_z, b_x, b_z

@ti.func
def panel_source_velocity_ti(x1, z1, x2, z2, xi, zi):
    t_x, t_z, n_x, n_z, x, z, d, r1, r2, theta1, theta2 = panel_info_ti(x1, z1, x2, z2, xi, zi)
    a_x = 0.0; a_z = 0.0
    
    if d >= 1e-15:
        logr1 = 0.0; logr2 = 0.0
        if r1 >= 1e-9: 
            logr1 = ti.math.log(r1)
        else: 
            theta1 = ti.math.pi
            theta2 = ti.math.pi
        if r2 >= 1e-9: 
            logr2 = ti.math.log(r2)
        else: 
            theta1 = 0.0
            theta2 = 0.0
            
        pi2 = 2.0 * ti.math.pi
        u = (logr1 - logr2) / pi2
        w = (theta2 - theta1) / pi2
        
        a_x = u*t_x + w*n_x
        a_z = u*t_z + w*n_z
        
    return a_x, a_z

@ti.data_oriented
class TaichiFlowField:
    def __init__(self, res_x=200, res_z=200):
        self.res_x = res_x
        self.res_z = res_z
        self.V_field = ti.Vector.field(2, dtype=ti.f32, shape=(self.res_x, self.res_z))

    def solve(self, M_foil, alpha_deg, grid_bounds=(-0.5, 1.5, -1.0, 1.0)):
        X_np = M_foil.foil.x.T.astype(np.float32) # (N, 2)
        G_np = M_foil.isol.gam.flatten().astype(np.float32) # (N,)
        
        N = X_np.shape[0]
        
        X_ti = ti.Vector.field(2, dtype=ti.f32, shape=N)
        G_ti = ti.field(dtype=ti.f32, shape=N)
        X_ti.from_numpy(X_np)
        G_ti.from_numpy(G_np)
        
        viscous_flag = 1 if M_foil.oper.viscous else 0
        Nw = M_foil.wake.N if M_foil.oper.viscous else 0
        N_sig = N - 1 + max(0, Nw - 1)
        
        Xw_ti = ti.Vector.field(2, dtype=ti.f32, shape=max(1, Nw))
        Sig_ti = ti.field(dtype=ti.f32, shape=max(1, N_sig))
        
        if viscous_flag == 1 and Nw > 0:
            Xw_ti.from_numpy(M_foil.wake.x.T.astype(np.float32))
            m = M_foil.glob.U[1, :] * M_foil.glob.U[3, :] # ds * ue
            sigma = M_foil.vsol.sigma_m @ m
            Sig_ti.from_numpy(sigma.astype(np.float32))
        
        # TE Info
        from mfoil import TE_info
        _, _, _, tcp_val, tdp_val = TE_info(M_foil.foil.x)
        
        Vinf = 1.0
        alpha_rad = np.radians(alpha_deg)
        vinf_x = Vinf * np.cos(alpha_rad)
        vinf_z = Vinf * np.sin(alpha_rad)

        @ti.kernel
        def compute_field():
            xmin, xmax = float(grid_bounds[0]), float(grid_bounds[1])
            zmin, zmax = float(grid_bounds[2]), float(grid_bounds[3])
            
            dx = (xmax - xmin) / ti.cast(self.res_x - 1, ti.f32)
            dz = (zmax - zmin) / ti.cast(self.res_z - 1, ti.f32)
            
            tcp = float(tcp_val)
            tdp = float(tdp_val)
            
            for i, j in self.V_field:
                px = xmin + ti.cast(i, ti.f32) * dx
                pz = zmin + ti.cast(j, ti.f32) * dz
                
                vx = float(vinf_x)
                vz = float(vinf_z)
                
                for p in range(N - 1):
                    p1_x = X_ti[p][0]; p1_z = X_ti[p][1]
                    p2_x = X_ti[p + 1][0]; p2_z = X_ti[p + 1][1]
                    ax, az, bx, bz = panel_vortex_velocity_ti(p1_x, p1_z, p2_x, p2_z, px, pz)
                    vx += ax * G_ti[p] + bx * G_ti[p + 1]
                    vz += az * G_ti[p] + bz * G_ti[p + 1]
                    
                    if viscous_flag == 1:
                        sax, saz = panel_source_velocity_ti(p1_x, p1_z, p2_x, p2_z, px, pz)
                        vx += sax * Sig_ti[p]
                        vz += saz * Sig_ti[p]
                        
                if viscous_flag == 1 and Nw > 0:
                    for p in range(Nw - 1):
                        p1_x = Xw_ti[p][0]; p1_z = Xw_ti[p][1]
                        p2_x = Xw_ti[p + 1][0]; p2_z = Xw_ti[p + 1][1]
                        sax, saz = panel_source_velocity_ti(p1_x, p1_z, p2_x, p2_z, px, pz)
                        vx += sax * Sig_ti[N - 1 + p]
                        vz += saz * Sig_ti[N - 1 + p]
                    
                te1_x = X_ti[N - 1][0]; te1_z = X_ti[N - 1][1]
                te2_x = X_ti[0][0]; te2_z = X_ti[0][1]
                sax, saz = panel_source_velocity_ti(te1_x, te1_z, te2_x, te2_z, px, pz)
                vx += sax * (-0.5 * tcp) * G_ti[0] + sax * (0.5 * tcp) * G_ti[N - 1]
                vz += saz * (-0.5 * tcp) * G_ti[0] + saz * (0.5 * tcp) * G_ti[N - 1]
                
                vax, vaz, vbx, vbz = panel_vortex_velocity_ti(te1_x, te1_z, te2_x, te2_z, px, pz)
                vx += (vax + vbx) * (0.5 * tdp) * G_ti[0] + (vax + vbx) * (-0.5 * tdp) * G_ti[N - 1]
                vz += (vaz + vbz) * (0.5 * tdp) * G_ti[0] + (vaz + vbz) * (-0.5 * tdp) * G_ti[N - 1]
                
                self.V_field[i, j] = [vx, vz]

        compute_field()
        v_res = self.V_field.to_numpy() # (res_x, res_z, 2)
        
        x = np.linspace(grid_bounds[0], grid_bounds[1], self.res_x)
        z = np.linspace(grid_bounds[2], grid_bounds[3], self.res_z)
        X, Z = np.meshgrid(x, z, indexing='xy')
        
        return X, Z, v_res[:,:,0].T, v_res[:,:,1].T
