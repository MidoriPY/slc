using Random
using Base
using Statistics
using LinearAlgebra
using IJulia
using Serialization
using Threads

# include("../modules/h_structurefactor.jl")
include("slc_functions.jl")


#*** MAIN function to run MC (annealing, thermalization, MC measurements :calculate ground state energy, magnetization, spec. heat, susceptibility etc.) ***
function calcul(ParamHam, ParamSys, config, Sq_list, T_ann) #input: configuration of NxN spin system, number of steps until thermalization, number of error runs,i.e. trial runs, autocorrelation time.
    
    Energies = Float64[] #! todo put back energy measurements to #MEASUREMENT section (i.e. only during MC steps)
    Energies_squared = Float64[]

    
    E =0.0
    #*** ANNEALING ***
    beta_ann = 1 ./T_ann 
    
    t1 = time()
    for n in 1:ParamSys.annsteps
        for m in 1:ParamSys.stepsperann
            config,Sq_list, E = slc_functions.MC_step(ParamHam, ParamSys, Sq_list, config, beta_ann[n],E)
            
            # append!(Energies,E)
        end
    end
    elapsed_time1 = time() - t1;
    println("elapsed time for annealing: ", elapsed_time1)


    #*** #THERMALIZATION ***
    beta = 1 / ParamSys.T
    t1 = time()
    for i in 1:ParamSys.eqsteps
        config,Sq_list, E = slc_functions.MC_step(ParamHam, ParamSys,Sq_list, config, beta,E)   
        
        # @assert isapprox(config, copy_oldconfig,atol=1e-10)
    end
    elapsed_time1 = time() - t1;
    println("elapsed time for thermalization: ", elapsed_time1)

    
    #*** MEASUREMENT ***

    magnetization, magnetization_sq = Float64[], Float64[]
    
    t1  =time()
    k=1::Int64
    stepsize = ParamSys.mcstep/ParamSys.sn
    for i in 1:ParamSys.mcstep
    
            config,Sq_list, E = slc_functions.MC_step(ParamHam, ParamSys, Sq_list,config, beta,E)
            
        
            if i == k*stepsize #ParamSys.mcstep-10^4 + k*1000
                

                En = En_squared = 0.0
                Mag, Mag_squared = 0.0, 0.0
                for l in 1:ParamSys.mn
                    
                    config,Sq_list, E = slc_functions.MC_step(ParamHam, ParamSys,Sq_list, config, beta,E)  
    
                    En += E#/ParamSys.N^3
                    En_squared += (E/ParamSys.N^3)^2
                    
                    m = slc_functions.magnetization_per_site(config, ParamSys, ParamHam)
                    Mag += m
                    Mag_squared += m^2
        
                end
    

                #*** store observables ***
                Energy = En/ParamSys.mn
                Energy_squared = En_squared/ParamSys.mn
                append!(Energies,Energy)
                append!(Energies_squared,Energy_squared)
                append!(magnetization, Mag/ParamSys.mn)
                append!(magnetization_sq, Mag_squared/ParamSys.mn)
            

                i += ParamSys.mn
                k += 1
            end                   
        
    end 

    elapsed_time1 = time() - t1;
    println("elapsed time for mc energy+magn measurement: ", elapsed_time1)
    
    t1 = time()
    for j in 1:100 #! change this to 100
        config,Sq_list,E = slc_functions.MC_step(ParamHam, ParamSys, Sq_list, config, beta,E) 
        
    end

    mean_u, mean_v, mean_w = zeros(ParamSys.N,ParamSys.N,ParamSys.N),zeros(ParamSys.N,ParamSys.N,ParamSys.N),zeros(ParamSys.N,ParamSys.N,ParamSys.N)
    sf_mean = zeros(ParamSys.N,ParamSys.N,ParamSys.N)
    mq_list_mean = zeros(3)
    for j in 1:ParamSys.lm
        config,Sq_list ,E= slc_functions.MC_step(ParamHam, ParamSys, Sq_list,config, beta,E) 
    
        #extract spin config
        mean_u += config[:,:,:,1]
        mean_v += config[:,:,:,2]
        mean_w += config[:,:,:,3]

        ssf_func, mq_list = slc_functions.structurefactor_mq(ParamHam,ParamSys,config) #structurefactor(ParamSys,config)
        sf_mean .+=  ssf_func#real.(structurefactor(ParamSys,config))
        mq_list_mean .+= sqrt.(mq_list/ParamSys.N)   
    end

    spin = zeros(ParamSys.N,ParamSys.N,ParamSys.N,3)
    spin[:,:,:,1] = mean_u ./ ParamSys.lm
    spin[:,:,:,2] = mean_v ./ ParamSys.lm
    spin[:,:,:,3] = mean_w ./ ParamSys.lm

    for m in 1:ParamSys.N
        for n in 1:ParamSys.N
            for o in 1:ParamSys.N
                spin[m,n,o,:] ./= norm(spin[m,n,o,:])
            end
        end
    end

    
    sf_mean ./= (ParamSys.lm*(ParamSys.N^3))
    mq_list_mean ./= ParamSys.lm
    elapsed_time1 = time() - t1;
    println("elapsed time for mean config and ssf : ", elapsed_time1)
    return spin, Energies,Energies_squared, sf_mean, magnetization, magnetization_sq, mq_list_mean 
end





mutable struct paramham
    Q::Float64
    J::Float64
    K::Float64
    D::Float64
    h::Vector{Float64}
    q_list::Matrix{Float64}
end

mutable struct paramsys
    N::Int64 #System size
    T::Float64 #Temperature
    mcstep::Int64 
    eqsteps::Int64    
    stepsperann::Int64
    annsteps::Int64
    mn::Int64
    sn::Int64
    lm::Int64
    tnum::Int64
end

function initialize()
    J = 1.0
    K = 0.7
    D = 0.3
    h = zeros(3)  #*ones(3)/sqrt(3) #
    # magnetic field array
    hsteps = 10
    h_array = range(0.0, stop=3.0, length=hsteps)#range(0.01, stop=3.0, length=hsteps)
    N = 8
    T = 0.0001 #Temperature for Monte Carlo

    mcstep = 10^3#3 #* Monte Carlo steps
    eqsteps = 1#3 #* thermalisation steps
    stepsperann = 10^4#0^2  #* steps per annealing temperature step 
    annsteps =20    #* annealing steps
    mn = 10          #* 100 mean number
    sn = 10     #* ^2 #number of snapshots(measurements)
    lm = 50
    tnum = 10

    #ANNEALING
    Tinit_ann = 1.0
    Tend_ann = 0.0001
    Tarray = LinRange(Tinit_ann,Tend_ann,annsteps)
    T_ann = zeros(annsteps)

    alpha= exp(log(Tend_ann/Tinit_ann)/annsteps)::Float64

    T_ann[end] = Tinit_ann
    #Tarray_ann = LinRange(Tinit_ann,Tend_ann,annsteps)
    for i in 1:annsteps-1
        T_ann[end-i] = T_ann[end-i+1]*alpha
    end
    T_ann = reverse(T_ann)
    

    @threads for i in 1:10
        Random.seed!(50*i)
        config =  rand(N,N,N,3)# # #helicalstate_test(N,Q) threeQ_HL(N, 0.0) 
        for i in 1:N
            for j in 1:N
                for l in 1:N
                    config[i,j,l,:] = config[i,j,l,:]/norm(config[i,j,l,:])
                end
            end
        end
    
        Q = pi/4
        q_list = Matrix{Float64}(I, 3, 3)*Q
        Sq_list = zeros(ComplexF64, 3, 3)


        

        for i in 1:3

            Sq_list[i,:] = slc_functions.S_q( q_list[i,:], config, N)
        end


        results = []

        #set all parameters
        ParamSys = paramsys(N,T, mcstep, eqsteps, stepsperann, annsteps, mn, sn, lm, tnum) #, A, m, shift
        ParamHam = paramham(Q,J,K,D,h, q_list)
                
        # #MAIN FUNCTION
        println(" field is now : ", 0)
        newconfig,E,E_sq,sf, magn, magn_sq, mq_list = calcul(ParamHam, ParamSys, config, Sq_list, T_ann)
        # append!(results, [newconfig,E,E_sq,sf, magn, magn_sq])
        open("data/inf_3Q_sd_$(i)_[001]_h_$(0.0).jls", "w") do file
            serialize(file,[newconfig,E,E_sq,sf, magn, magn_sq,mq_list, 0.0])
        end
        # #* for field scans
        #annealing at each h step

        Tinit_ann = 0.01
        Tend_ann = 0.0001
        Tarray = LinRange(Tinit_ann,Tend_ann,annsteps)
        T_ann = zeros(annsteps)

        alpha= exp(log(Tend_ann/Tinit_ann)/annsteps)::Float64

        T_ann[end] = Tinit_ann::Float64
        #Tarray_ann = LinRange(Tinit_ann,Tend_ann,annsteps)
        for i in 1:annsteps-1
            T_ann[end-i] = T_ann[end-i+1]*alpha
        end
        T_ann = reverse(T_ann)

        #* uncomment for field scan
        # for i in 2:hsteps
        #     println(" field is now : ", h_array[i])
        #     hnew = [0.0,0.0,h_array[i]]
        #     ParamHam = paramham(Q,J,K,D,hnew, q_list)
        #     # ParamSys = paramsys(N,T, mcstep, eqsteps, stepsperann, annsteps, mn, sn, lm, tnum) #, A, m, shift
        #     newconfig,E,E_sq,sf, magn, magn_sq, mq_list = calcul(ParamHam, ParamSys, config, Sq_list, T_ann)
        #     # append!(results, [newconfig,E,E_sq,sf, magn, magn_sq,mq_list, h_array[i]])
        #     open("data/inf_3Q_test_seed2_[001]_h_$(round(h_array[i],digits= 2)).jls", "w") do file
        #         serialize(file,[newconfig,E,E_sq,sf, magn, magn_sq,mq_list, h_array[i]])
        #     end
            
        # end

    end

end
    
initialize()
