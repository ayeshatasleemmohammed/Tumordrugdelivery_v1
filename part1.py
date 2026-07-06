##Assuming Every particle reaches the tumor and releases the drug
##comparing different cases of how drug reaches the tumor affected cell
##checking efficiency of each case and plotting it w.r.t time
##visualising trajectories of drug particles in x-y directions w.r.t time
import numpy as np
import matplotlib.pyplot as plt
no_of_particles = 1000
steps = 1000
tumor_center = np.array([30.0,0])
tumor_radius = 3.0
blood_drift = np.array([0.1,0])
def efficiency(case_type):
    np.random.seed(42)
    is_bound = np.zeros(no_of_particles,dtype = bool)
    position = np.zeros([no_of_particles,2])
    position[:,0] = np.random.uniform(-5,-3,no_of_particles)
    position[:,1] = np.random.uniform(-2,2,no_of_particles)
    particle_size = np.ones(no_of_particles)*40
    diffusion_strength = np.random.uniform(0,0.4,no_of_particles)
    attraction_strength = np.random.uniform(0,0.3,no_of_particles)
    attraction_strength = attraction_strength.reshape(-1,1)
    diffusion_strength  =  diffusion_strength.reshape(-1,1)
    captured_history = []
    trajectory = np.zeros((steps,no_of_particles,2))
    for t in range(steps):
        direction_to_tumor = tumor_center-position
        distance = np.linalg.norm(direction_to_tumor,axis = 1,keepdims = True)
        distance[distance == 0] = 1e-8
        unit_direction = direction_to_tumor/distance
        pressure_strength = np.random.uniform(0, 0.08, no_of_particles).reshape(-1,1)
        pressure_force = -unit_direction * pressure_strength
        if(case_type == 'case1'):
             brownian_movement = np.random.normal(0,diffusion_strength,(no_of_particles,2))
             total_movement = brownian_movement+blood_drift
             position += total_movement
        if(case_type=='case2'):
              brownian_movement = np.random.normal(0,diffusion_strength,(no_of_particles,2))
              total_movement = brownian_movement+(pressure_force)+blood_drift
              total_movement[is_bound]=[0.0,0.0]
              position += total_movement
        if(case_type=='case3'):
                brownian_movement = np.random.normal(0,diffusion_strength,(no_of_particles,2))
                total_movement = brownian_movement+unit_direction*attraction_strength+blood_drift
                position += total_movement
        if(case_type== 'case4'):
             dynamic_diffusion = np.where(particle_size==5.0,0.6,0.3)
             scalar_matrix = dynamic_diffusion.reshape(-1,1)
             active_pressure = np.where(particle_size.reshape(-1,1)==5.0,0.0,unit_direction*attraction_strength)
             brownian_movement = np.random.normal(0,scalar_matrix,(no_of_particles,2))
             total_movement = brownian_movement+(active_pressure)+blood_drift
             position += total_movement
        if(case_type=='case5'):
                        brownian_movement = np.random.normal(0,diffusion_strength,(no_of_particles,2))
                        total_movement = brownian_movement+unit_direction*attraction_strength+blood_drift+pressure_force
                        position += total_movement
        if(case_type== 'case6'):
                    dynamic_diffusion = np.where(particle_size==5.0,0.6,0.3)
                    scalar_matrix = dynamic_diffusion.reshape(-1,1)
                    active_pressure = np.where(particle_size.reshape(-1,1)==5.0,0.0,unit_direction*attraction_strength)
                    brownian_movement = np.random.normal(0,scalar_matrix,(no_of_particles,2))
                    total_movement = brownian_movement+(active_pressure)+blood_drift+pressure_force
                    position += total_movement
        trajectory[t]=position
        dis_fro_tumor = np.linalg.norm(position-tumor_center,axis = 1)
        capture = dis_fro_tumor<=tumor_radius
        if(case_type== 'case4'or case_type == 'case6'):
             particle_size[capture]=5.0
        is_bound = is_bound | capture
        efficiency = np.sum(is_bound)/no_of_particles
        captured_history.append(efficiency)
    return captured_history,trajectory
case1_efficiency,case1_trajectory=efficiency("case1")
case2_efficiency,case2_trajectory = efficiency("case2")
case3_efficiency,case3_trajectory = efficiency("case3")
case4_efficiency ,case4_trajectory= efficiency("case4")
case5_efficiency ,case5_trajectory= efficiency("case5")
case6_efficiency ,case6_trajectory= efficiency("case6")
print("X min:", np.min(case4_trajectory[:,:,0]))
print("X max:", np.max(case4_trajectory[:,:,0]))
print("Y min:", np.min(case4_trajectory[:,:,1]))
print("Y max:", np.max(case4_trajectory[:,:,1]))
print(case4_trajectory[0,:5,:])
print(case4_trajectory[-1,:5,:])
plt.figure(figsize=(10,6))
plt.plot(case1_efficiency,label = "passive diffusion",color='green')
plt.plot(case2_efficiency,label = "case of particles having same size",color='blue')
plt.plot(case3_efficiency,label = "Targeted drug delivery",color='yellow')
plt.plot(case4_efficiency,label = "case of particles which can break ",color='red')
plt.plot(case5_efficiency,label = "Targeted drug delivery+pressure force",color='pink')
plt.plot(case6_efficiency,label = "case of particles which can break+pressure force ",color='orange')
plt.xlabel("Time")
plt.ylabel("Efficiency")
plt.legend()
plt.grid(True,linestyle =":")
plt.show()
plt.figure(figsize=(8,8))
plt.scatter(30.0,0.0,color= 'red',s=200,marker = 'x')
plt.scatter(case2_trajectory[0,:100,0],case2_trajectory[0,:100,1],s=20,label = 'injecting point',color = 'green')
plt.scatter(case2_trajectory[-1,:100,0],case2_trajectory[-1,:100,1],s=20,label = 'ending  point',color = 'black')
for i in range(100):
     plt.plot(case2_trajectory[:,i,0],case2_trajectory[:,i,1],color = 'pink',alpha = 0.07)
circle = plt.Circle(
    tumor_center,
    tumor_radius,
    fill=False,
    color='red',
    linewidth = 4,
    label = 'Tumor'
)

plt.gca().add_patch(circle)
plt.xlim(-10,110)
plt.ylim(-30,30)
plt.title('trajectory of particles')
plt.legend()
plt.grid(True)
plt.show()
