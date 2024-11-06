module slc_functions
    using Random
    using Base
    using Statistics
    using LinearAlgebra
    using IJulia

    #create meshgrid 
    function meshgrid(x, y)
        X = [i for i in x, j in 1:length(y)]
        Y = [j for i in 1:length(x), j in y]
        return X, Y
    end

    #Fourier transform of S (real space spin configuration) at q vector q
    function S_q( q, S, N) #k vector where want to be evaluated, S= spinsystem NxN in 3 dim Sx,Sy,Sz
        S_k =zeros(ComplexF64,3)
        for i in 1:N
            for j in 1:N
                for l in 1:N
                    r = [i,j,l]
                    S_k += S[i,j,l,:].*exp(-im*q'r)
                end
            end
        end
    
        return (S_k)/sqrt(N^3)
    end
    #*******************************************

    #**** spin configuration ansatz functions  ****

    #Ansatz for 4Q HL state 
    function quadrupleQstate(ParamSys) #input size length
            
        e0 = zeros(4,3)
        e0[1,:] = [1.0,0.0,1.0]/sqrt(2)
        e0[2,:] = [-1.0,0.0,1.0]/sqrt(2)
        e0[3,:] = [0.0,1.0,-1.0]/sqrt(2)
        e0[4,:] = [0.0,-1.0,-1.0]/sqrt(2)


        e1 = zeros(4,3)
        e2 =zeros(4,3)
        Q = 2.0*pi/6.0
        Q1 = e0[1,:]*Q*sqrt(2)
        Q2 = e0[2,:]*Q*sqrt(2)
        Q3 = e0[3,:]*Q*sqrt(2)
        Q4 = e0[4,:]*Q*sqrt(2)

        x =[1.0,0.0,0.0]
        y =[0.0,1.0,0.0]
        z =[0.0,0.0,1.0]


        for l in  1:4
            ee =cross(z,e0[l,:])
            e1[l,:] = ee/norm(ee)
            e2[l,:] =cross(e0[l,:], e1[l,:])
        end

        a =ones(4)

        S =zeros((ParamSys.N,ParamSys.N,ParamSys.N,3))
        for i in 1:ParamSys.N
            for j in  1:ParamSys.N
                for k in 1:ParamSys.N
                    r = [i+0.5,j+0.5,k+0.5]
                    Q1r =dot(Q1,r)  #+pi/3
                    Q2r =dot(Q2,r)  #+pi/3
                    Q3r =dot(Q3,r)  #+pi/3
                    Q4r =dot(Q4,r)  #+pi/3
                    
                    Qr =[Q1r,Q2r,Q3r,Q4r]               
                    for l in  1:4
                        S1 = e1[l,:]*cos(Qr[l])
                        S2 = e2[l,:]*sin(Qr[l])

                        S[i,j,k,:] += ( S1+S2 ) 
                    end 
                    S[i,j,k,:] /= norm( S[i,j,k,:])
                    
                end
            end
        end
        
        return S

    end

    # ansatz for 3Q HL state
    function threeQ_HL(N, m)
        e0 = zeros(3,3)
        e0[1,1] = 1.0
        e0[2,2] = 1.0
        e0[3,3] = 1.0
        Q = pi/4.0
        phi = pi/8.0
        e1 = zeros(3,3)
        e2 = zeros(3,3)
        e3 = [[0,1,0] [0,0,1] [1,0,0]]
        Qvectors = zeros(3,3)
        for i in 1:3
            e1[i,:] = cross(e0[i,:], e3[i,:])
            e2[i,:] = cross(e0[i,:], e1[i,:])
            Qvectors[i,:] = e0[i,:]*Q
        end
        S = zeros(N,N,N,3)
        for i in 1:N
            for j in 1:N
                for k in 1:N
                    r = [i, j, k]
                    # Qr = [dot(Q*e0[1,:], r), dot(Q*e0[2,:], r), dot(Q*e0[3,:], r)]
                    for l in 1:3
                        S1 = e1[l,:]*cos(dot(Qvectors[l,:],r) + phi)  #Qr[l]
                        S2 = e2[l,:]*sin(dot(Qvectors[l,:],r) + phi)
                        S[i,j,k,:] += S1 + S2
                    end
                    S[i,j,k,:] /= norm(S[i,j,k,:])
                end
            end
        end
        return S
    end

    #ansatz for helical state
    function helicalstate(N,Q)
        S = zeros((N,N,N,3))
    
        Q1 = [1.0,1.0,1.0].*Q
        z = [1.0,1.0,-2.0]/sqrt(6)
        e0 =[1.0,1.0,1.0]/sqrt(3)
        e1 = cross(z,e0)
        for i in 1:N
            for j in 1:N
                for k in 1:N
                    r = [i+0.5,j+0.5,k+0.5]
                    Q1r = dot(Q1,r)
            
                    S1 = -z.*cos(Q1r) 
                    S2 =e1.*sin(Q1r)
                    S[i,j,k,:] += ( S1+S2 ) 
                    S[i,j,k,:] /= norm( S[i,j,k,:])
                end
            end
        end
        return S
    end
    #*******************************************

    #****  Hamiltonian functions  ****

    function get_JHami(sq_list, ParamHam) 
        E=0.0
        for i in 1:size(ParamHam.q_list)[1]

            # Jq = J_q(ParamHam.q_list[i,:],ParamHam.J,ParamSys.N)
            E +=(sq_list[i]'sq_list[i])
        end
        return  ParamHam.J*E
    end

    function get_biquadratic(sq_list, ParamHam, ParamSys) 
        E=0.0
        for i in 1:size(ParamHam.q_list)[1]
            E += (sq_list[i]'sq_list[i])^2
        end
        return ParamHam.K*E /ParamSys.N^3
    end

    function get_DM_3Q(sq_list, ParamHam) 
    
        E = (sq_list[1,2].*conj(sq_list[1,3]) - sq_list[1,3].*conj(sq_list[1,2]))
        E += (sq_list[2,3].*conj(sq_list[2,1]) - sq_list[2,1].*conj(sq_list[2,3]))
        E += (sq_list[3,1].*conj(sq_list[3,2]) - sq_list[3,2].*conj(sq_list[3,1]))
    
        return im*E*ParamHam.D
    end

    # function get_DM_1Q(sq_list, ParamHam, ParamSys) 
    #     E=0.0
    #     D = copy(ParamHam.D)
    #     E += (sq_list[2]*conj(sq_list[3]) - sq_list[3]*conj(sq_list[2]))
    #     E += (sq_list[3]*conj(sq_list[1]) - sq_list[1]*conj(sq_list[3]))
    #     E += (sq_list[1]*conj(sq_list[2]) - sq_list[2]*conj(sq_list[1]))

        
    #     return im*E*D
    # end


    function get_h(config, ParamHam, ParamSys) 
        E=0.0
        for i in 1:ParamSys.N
            for j in 1:ParamSys.N
                for l in 1:ParamSys.N
                    E += dot(ParamHam.h,config[i,j,l,:])
                end
            end
        end
        return E/(3*ParamSys.N^3)
    end
    #*******************************************

    #**** Monte Carlo spin update function ****

    function MC_step( ParamHam, ParamSys,Sq_list, config, beta, E) # q_list,Sq_list,beta)  #ParamHam, ParamSys, config, beta_ann[n])
        for i in 1:ParamSys.N
            for j in 1:ParamSys.N
                for k in 1:ParamSys.N
                
                    a,b,c  = rand(1:ParamSys.N), rand(1:ParamSys.N), rand(1:ParamSys.N)
                    
                    old_s = config[a,b,c,:]
                    U1, V1 = rand(),rand() 
                    new_ang = [acos(1-2*V1),2*pi*U1]
                    new_s = [sin(new_ang[1])*cos(new_ang[2]), sin(new_ang[1])*sin(new_ang[2]), cos(new_ang[1])]
                    newconfig = copy(config)
                    newconfig[a,b,c,:] = copy(new_s)
                
                    del_E, newE, newSq_list = getEdiff( newconfig, old_s, new_s,a,b,c, Sq_list, ParamHam, ParamSys, E) 
            
                    if rand() < min(1, exp(-real(del_E)*beta))
                        # println("------ accepted  ------")
                        config[a,b,c,:] = copy(new_s) 
                        Sq_list = copy(newSq_list)
                        E = copy(newE)
                    end
                end 
            end
        end
        return config, Sq_list, E
    end

    # compare energies, evaluate energy difference
    function getEdiff(newconfig, old_s, new_s,a,c,b, sq_list, ParamHam, ParamSys, E ) #, newconfig
        r = [a,b,c]

        # oldE = wholeHami(config, sq_list, ParamHam, ParamSys )#wholehamiltonian(config, a, b,c) in k space is fine
    
        new_Sq_list = zeros(ComplexF64, length(ParamHam.q_list),3)

        for i in 1:size(ParamHam.q_list)[1]
            new_Sq_list[i,:] = copy(sq_list[i,:]) .+ (new_s .- old_s ).*exp(-im*ParamHam.q_list[i,:]'r) /sqrt(ParamSys.N^3)
        end
    
        newE = wholeHami(newconfig, new_Sq_list, ParamHam, ParamSys)
        return newE - E, newE, new_Sq_list  # newE  energy of new config - energy of old config
    end


    function wholeHami(conf,sq_list, ParamHam, ParamSys)
        return  - get_h(conf, ParamHam, ParamSys)  - get_JHami(sq_list, ParamHam) + get_biquadratic(sq_list, ParamHam, ParamSys) - get_DM_3Q(sq_list, ParamHam)
    end


    #*******************************************
    #**** spin structure factor functions ****

    #Fourier transform of Sk*S-k at vector k
    function S_k(ParamSys, k,S) #k vector where want to be evaluated, S= spinsystem NxN in 3 dim Sx,Sy,Sz

        S_k,S_minusk =zeros(ComplexF64,3),zeros(ComplexF64,3)
        for i in 1:ParamSys.N
            for j in 1:ParamSys.N
                for l in 1:ParamSys.N
                    r = [i,j,l]
                    S_k += S[i,j,l,:].*exp(im*k'r)
                end
            end
        end

        return (S_k'S_k)/(ParamSys.N^3 )
    end

    # main spin structure factor function
    function structurefactor(ParamSys, system)

        kx, ky = range(0.0,2π,length=ParamSys.N+1),range(0.0,2π,length=ParamSys.N+1)
        aKX,aKY = meshgrid(kx,ky)
        ww, KZ = meshgrid(ky,aKY)
        KY, ww = meshgrid(aKX,aKY)
        aKZ, KX = meshgrid(kx,aKX)
        KY = KY[:,:,1:ParamSys.N]

        S = zeros(Float64,ParamSys.N,ParamSys.N,ParamSys.N)
        for i in 1:ParamSys.N
            for j in 1:ParamSys.N
                for l in 1:ParamSys.N
                    k = [KX[i,j,l], KY[i,j,l],KZ[i,j,l]]
                    S[i,j,l] = real(S_k(ParamSys, k,system))
                end
            end
        end
        return round.(S, digits=10)
    end

    #main spin structure factor function and m_q evaluation (magnetic moment with q)
    function structurefactor_mq(ParamHam, ParamSys, system)

        kx, ky = range(0.0,2π,length=ParamSys.N+1),range(0.0,2π,length=ParamSys.N+1)

        aKX,aKY = meshgrid(kx,ky)
        ww, KZ = meshgrid(ky,aKY)
        KY, ww = meshgrid(aKX,aKY)
        aKZ, KX = meshgrid(kx,aKX)
        KY = KY[:,:,1:ParamSys.N]

        S = zeros(Float64,ParamSys.N,ParamSys.N,ParamSys.N)
        
        for i in 1:ParamSys.N
            for j in 1:ParamSys.N
                for l in 1:ParamSys.N
                    k = [KX[i,j,l], KY[i,j,l],KZ[i,j,l]]
                    S[i,j,l] = real(S_k(ParamSys, k,system))
                end

            end
        end
        
        mq_list = Float64[]
        for k in 1:3 #!3Q case only, adjust for 4Q
            push!(mq_list, real(S_k(ParamSys, ParamHam.q_list[k,:],system)))
        end
        return round.(S, digits=10), mq_list

    end

    #*******************************************

    #****  observables functions  ****
    #compute magneization per spin spinsite

    function magnetization_per_site( config, ParamSys, ParamHam)
        m = 0.0
        # if ParamHam.h == [0.0,0.0,0.0]
        if isapprox(ParamHam.h, [0.0, 0.0, 0.0]; atol=1e-8)
            # unit_h = [0.0,0.0,0.0]
            return 0.0
        
        else
            unit_h = ParamHam.h/norm(ParamHam.h)
            for i in 1:ParamSys.N
                for j in 1:ParamSys.N
                    for l in 1:ParamSys.N
                        
                    
                        m += dot(config[i,j,l,:], unit_h)
                        # println("magnetization per site", m)
                    end
                end
            end
            return m/(ParamSys.N^3)
        end
    end

    #*******************************************

end