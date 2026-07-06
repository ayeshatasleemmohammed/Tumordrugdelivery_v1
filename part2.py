"""
Targeted Drug Delivery Simulation

This project compares four nanoparticle delivery strategies:
1. Passive transport
2. Tumor pressure effects
3. Targeted nanoparticles
4. pH-responsive shrinking nanoparticles

The model incorporates blood flow, Brownian diffusion,
tumor binding, immune clearance, and probabilistic drug release
to evaluate delivery efficiency under different assumptions.
"""
# Case 1: Passive nanoparticle transport driven by blood flow and diffusion.
# Case 2: Effect of tumor interstitial pressure opposing nanoparticle transport.
# Case 3: Targeted nanoparticles with enhanced tumor attraction and binding.
# Case 4: pH-responsive nanoparticles that shrink near the tumor, increasing diffusivity.
# Model Assumptions
# 1. Blood flow follows a steady parabolic velocity profile.
# 2. Tumor geometry is represented as a fixed circular region.
# 3. Binding probability is constant for a given nanoparticle type.
# 4. Immune clearance is modeled as a constant stochastic probability.
# 5. Particle diffusion follows Brownian motion.
# 6. Tumor targeting is represented by an attraction force toward the tumor center.
# 7. pH-responsive particles shrink when exposed to the tumor microenvironment.
# 8. Drug release is modeled probabilistically after successful tumor localization.
import numpy as np
import matplotlib.pyplot as plt
no_of_particles =1500
steps = 100
tumor_center = np.array([10.0e-3,0])
tumor_radius = 2.0e-3
def simulation(case_type):
    np.random.seed(42)
    capture = []
    position = np.zeros((no_of_particles,2))
    position[:,0]=np.random.uniform(-15.0e-3,tumor_center[0]-tumor_radius,no_of_particles)
    position[:,1]=np.random.uniform(-1.0e-3,1.0e-3,no_of_particles)
    particle_size = np.ones(no_of_particles)*40
    vmax = 1.0e-3
    r = 1.0e-3
    dt = 1.0 #time for each step
    blood_drift = np.zeros((no_of_particles,2))
    if (case_type == "case1"):
        attraction_strength = 0.0
        binding_probability = 0.2
    elif(case_type == "case2"):
        attraction_strength = 1.0e-6
        binding_probability = 0.1
    else:
        binding_probability = 0.8
        attraction_strength = 2.0e-4
    diffusion_real = 1.0e-11
    diffusion_strength = np.sqrt(2*diffusion_real*dt)
    is_bound = np.zeros(no_of_particles,dtype = bool)
    alive = np.ones(no_of_particles,dtype= bool)
    organ_bypass = 0.005
    drug_release_probability = 0.05
    immune_clearence_probability = 0.01 # PEG-coated particles
    #immune_clearence_probability = 0.1 #Uncoated particles
    is_drug_released = np.zeros(no_of_particles,dtype = bool)
    trajectory = np.zeros((steps,no_of_particles,2))
    acidic_time = np.zeros(no_of_particles)
    for t in range(steps):
        organ_bypass_drug = np.random.uniform(0,0.1,no_of_particles)
        probaility_of_drug = np.random.uniform(0,1,no_of_particles)
        probability_of_each_particle = np.random.uniform(0,1,no_of_particles)
        direction_to_tumor = tumor_center - position
        distance = np.linalg.norm(direction_to_tumor,axis = 1,keepdims = True)
        distance[distance==0]= 1e-8
        unit_direction = direction_to_tumor/distance
        pressure_strength = np.random.uniform(0,0.8e-3,no_of_particles).reshape(-1,1)
        pressure_force = -unit_direction*pressure_strength
        brownian_movement = np.random.normal(0,diffusion_strength,(no_of_particles,2))
        blood_drift[:,0] = vmax*(1-((position[:,1])**2/r**2))
        if(case_type == "case1"):
            total_movement = brownian_movement+blood_drift+unit_direction*attraction_strength
            total_movement[is_bound|~alive] = [0.0,0.0]
            position += total_movement
        elif(case_type == "case2"):
            total_movement = brownian_movement+blood_drift+pressure_force
            total_movement[is_bound|~alive] = [0.0,0.0]
            position += total_movement
        elif(case_type == "case3"):
            total_movement = brownian_movement+blood_drift+(unit_direction*attraction_strength)
            total_movement[is_bound|~alive] = [0.0,0.0]
            position += total_movement
        elif(case_type == "case5"):
                total_movement = brownian_movement+blood_drift+(unit_direction*attraction_strength)+pressure_force
                total_movement[is_bound|~alive] = [0.0,0.0]
                position += total_movement
        elif(case_type == "case6"): 
            dynamic_diffusion = np.where(particle_size.reshape(-1,1)==5.0,diffusion_strength*np.sqrt(8),diffusion_strength).reshape(-1,1)
            attraction_short = np.where(particle_size.reshape(-1,1)==5.0,0.0,attraction_strength)
            brownian_movement = np.random.normal(0,dynamic_diffusion,(no_of_particles,2))
            total_movement = brownian_movement+blood_drift+(unit_direction*attraction_short)+pressure_force
            total_movement[is_bound|~alive]=[0.0,0.0]
            position += total_movement       
        else:
            dynamic_diffusion = np.where(particle_size.reshape(-1,1)==5.0,diffusion_strength*np.sqrt(8),diffusion_strength).reshape(-1,1)
            attraction_short = np.where(particle_size.reshape(-1,1)==5.0,0.0,attraction_strength)
            brownian_movement = np.random.normal(0,dynamic_diffusion,(no_of_particles,2))
            total_movement = brownian_movement+blood_drift+(unit_direction*attraction_short)
            total_movement[is_bound|~alive]=[0.0,0.0]
            position += total_movement
        trajectory[t]=position
        dis_fro_tumor = np.linalg.norm(position - tumor_center,axis = 1)
        binding_roll = np.random.rand(no_of_particles)
        if(case_type == "case3" or case_type=="case5"or case_type=="case1"or case_type=="case2" ):
                    new_captured = (dis_fro_tumor<=tumor_radius)&(~is_bound)&(alive)
                    new_bound = (new_captured)&(binding_roll<binding_probability)
                    is_bound[new_bound]=True
                    wasted = (probability_of_each_particle<= immune_clearence_probability)&(alive)&(~is_bound)
                    alive[wasted]=False
                    bypassed = (organ_bypass_drug<= organ_bypass)&(dis_fro_tumor>tumor_radius)&alive&~is_bound
                    alive[bypassed] = False
                    is_bound[bypassed]=False
                    released = (dis_fro_tumor<=tumor_radius-1.0e-3)&(probaility_of_drug>=drug_release_probability)&(alive)&(~is_drug_released)&(is_bound)
                    is_drug_released[released]=True
                    efficiency = np.sum(is_drug_released)/no_of_particles
        elif(case_type == "case4" or case_type == "case6"):
                    new_captured = (dis_fro_tumor<=tumor_radius)&(~is_bound)&(alive)
                    new_bound = (new_captured)&(binding_roll<binding_probability)
                    is_bound[new_bound]=True
                    wasted = (probability_of_each_particle<= immune_clearence_probability)&(alive)&(~is_bound)
                    alive[wasted]=False
                    bypassed = (organ_bypass_drug<= organ_bypass)&(dis_fro_tumor>tumor_radius)&alive&~is_bound
                    alive[bypassed] = False
                    is_bound[bypassed]=False
                    acidic_zone = dis_fro_tumor<=(tumor_radius+1.0e-3)
                    acidic_time[acidic_zone]+=dt
                    acidic_time[~acidic_zone]=0.0
                    particle_size[acidic_time>5.0]= 5.0
                    released = (dis_fro_tumor<=tumor_radius-1.0e-3)&(probaility_of_drug>=drug_release_probability)&(alive)&(~is_drug_released)&(is_bound)
                    is_drug_released[released]=True
                    efficiency = np.sum(is_drug_released)/no_of_particles
        capture.append(efficiency)
    return capture,trajectory
captured_history1,trajectory_case1 = simulation("case1")
captured_history2,trajectory_case2 = simulation("case2")
captured_history3 ,trajectory_case3= simulation("case3")
captured_history4,trajectory_case4 = simulation("case4")
captured_history5,trajectory_case5 = simulation("case5")
captured_history6,trajectory_case6 = simulation("case6")
plt.figure(figsize=(10,6))
plt.plot(captured_history1,label = f"passive diffusion(case1),{captured_history1[-1]:.4f}",color = "green")
plt.plot(captured_history2,label = f"pressure force of tumor effect(case2),{captured_history2[-1]:.4f}",color = "pink")
plt.plot(captured_history3,label = f"Targeted Drug delivery(case3),{captured_history3[-1]:.4f}",color = "yellow")
plt.plot(captured_history4,label = f"shrinkage of particles(case4),{captured_history4[-1]:.4f}",color = "red")
plt.plot(captured_history5,label = f"Targeted Drug delivery+pressure force(case5),{captured_history5[-1]:.4f}",color = "orange")
plt.plot(captured_history6,label = f"shrinkage of particles+pressure force(case6),{captured_history6[-1]:.4f}",color = "blue")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True,linestyle = ":")
plt.show()
plt.figure(figsize=(10,6))
plt.scatter(trajectory_case6[0,:30,0],trajectory_case6[0,:30,1],s = 20,color = "green",label="point of injection")
plt.scatter(trajectory_case6[-1,:30,0],trajectory_case6[-1,:30,1],s = 20,color = "black",label="point of end")
for i in range(30):
     plt.plot(trajectory_case6[:,i,0],trajectory_case6[:,i,1],color ="pink",alpha = 0.5)
circle = plt.Circle(tumor_center,tumor_radius,fill = False,linewidth = 4,color = "red",label ="Tumor")
plt.gca().add_patch(circle)
plt.xlim(-20e-3,15e-3)
plt.ylim(-5.0e-3,5.0e-3)
plt.legend()
plt.grid(True)
plt.show()
