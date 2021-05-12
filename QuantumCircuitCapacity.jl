
"""
Program to calculate gradient and QFIM of quantum circuits.

Based on Julia and Yao.

Tobias Haug @ Imperial College London

t.haug@imperial.ac.uk

"""

using Yao
using LinearAlgebra, Plots

using Statistics
using Random




##warn: fidelity function of yao is lacking ^2 for some reason. Make sure to add yourself


@const_gate SQISWAP = ComplexF64[1 0 0 0;0 1/sqrt(2) -1/sqrt(2)*1.0im 0;0 -1/sqrt(2)*1.0im 1/sqrt(2) 0;0 0 0 1]

@const_gate InvSQISWAP = ComplexF64[1 0 0 0;0 1/sqrt(2) 1/sqrt(2)*1.0im 0;0 1/sqrt(2)*1.0im 1/sqrt(2) 0;0 0 0 1]


@const_gate RyHalf = mat(Ry(pi/2))
@const_gate InvRyHalf = mat(Ry(-pi/2))

@const_gate SqrtH = mat(Ry(pi/4))

@const_gate InvSqrtH = mat(Ry(-pi/4))




#NOTE: Yao internal function for daggered is very slow. Thus, we construct the conjugated circuit ourselves

#Yao: zero_state(n_qubits) |> circuit applies circuit onto state immdeiatly
# zero_state(n_qubits) => circuit delays application, is evaluated only when e.g. expecation values are needed
#use => when calculating gradients via expect'()


##set parameters for calculation
n_qubits=4 #number qubits
depth=10##2^(div(n_qubits,2)) #depth of circuit

rng_seed=1 #seed for random generator

circuit_type=0 #0: CPHASE with random single qubit rotations, 1: CNOT with random single qubit rotations 2: Sqrt[iswap] with random single qubit rotations

choose_param_type=0#0: random parameters, 1: fix to 0

cache_entangler=false #whether to cache entangling gates. Causes issues for constant gates, also seems to be slower except for cnot gates. Suggest to keep false.


epsilon_inv=10^-12#threshold of inversion of Gtensor, e.g. limits the size of the larget inverted eigenvalue. Use this to filter out eigenvalues that are actually zero, but rounded up due to numerical rounding errors
epsilon_inv_replace=0#1/epsilon_inv#if threshold is hit, replace inverse with this

use_yao_adjoint=false #set this false, else it will be very slow.. But take care to implement daggered properly first!

starttime=time()

println("")


function get_pauli_circuit(curr_depth,curr_qubit,circuit_spec,n_qubits)
    """
    get pauli operator from circuit definition. Required for the analytic gradient for the QFIM
    """
    element=circuit_spec[(curr_depth-1)*n_qubits+curr_qubit]

    if element==0
        return I2
    elseif element==1
        return X
    elseif element==2
        return Y
    elseif element==3
        return Z
    elseif element==5
        return 1/sqrt(2)*(X+Y)
    elseif element==6
        return 1/sqrt(2)*(Z+X)
    else
        error("not defined ",element)
    end

end

function get_pauli_op(pauli_string)
    """
    turns pauli string into operator
    """
    n_qubits=length(pauli_string)
    pauli_circuit = chain(n_qubits)
    for i in 1:length(pauli_string)
        if pauli_string[i]!=0
            if pauli_string[i]==1
                push!(pauli_circuit, put(i=>X))
            elseif pauli_string[i]==2
                push!(pauli_circuit, put(i=>Y))
            elseif pauli_string[i]==3
                push!(pauli_circuit, put(i=>Z))
            end
        end
    end

    return pauli_circuit
end

#creates entangling gate
#daggered whether to create adjoint of it or not
function entangler(entanglingGate,n_qubits,daggered)
    if daggered==false
        ent=chain(n_qubits,if type==0; control(ctrl, target=>Z) elseif type==1;control(ctrl, target=>X) elseif type==2;put((ctrl,target)=>SQISWAP)   end for (ctrl,target,type) in entanglingGate)

    else#make sure its properly daggered, i.e. also reverse order and conjugate
        ent=chain(n_qubits,if type==0; control(ctrl, target=>Z) elseif type==1;control(ctrl, target=>X) elseif type==2;put((ctrl,target)=>InvSQISWAP)  end for (ctrl,target,type) in entanglingGate[length(entanglingGate):-1:1])
    end
    return ent
end



function build_circuit(n_qubits, nlayers, entanglingGates,circuit_spec,daggered=false)
    """
    construct circuit according to specifications
    """
    circuit = chain(n_qubits)
    param_type=[]

    column=1
    if(daggered==false)
        counter=0
        #H squareroot, use matblock(mat()) to fix parameters. There may be better way to do this
        push!(circuit, chain(n_qubits, put(i=>SqrtH) for i = 1:n_qubits))
        #push!(circuit, cache(chain(n_qubits, put(i=>H) for i = 1:n_qubits)))
        countEntanglingLayers=0
        for i in 1:nlayers
            countEntanglingLayers+=1
            push!(circuit, chain(n_qubits, put(p=>chain(RotationGate(get_pauli_circuit(i,p,circuit_spec,n_qubits),0))) for p = 1:n_qubits))

            counter+=n_qubits

            entgl=entangler(entanglingGates[countEntanglingLayers],n_qubits,daggered)

            #whether to cache entanglers. Seems to cause issues for constant gates
            if cache_entangler==true
                push!(circuit, cache(entgl))
            else
                push!(circuit, entgl)
            end

        end
    else #daggered version of circuit
        counter=length(circuit_spec)

        countEntanglingLayers=nlayers+1
        for i in nlayers:-1:1
            countEntanglingLayers-=1
            #do the daggered version of the entangler
            entgl=entangler(entanglingGates[countEntanglingLayers],n_qubits,daggered)

            if cache_entangler==true
                push!(circuit, cache(entgl))
            else
                push!(circuit, entgl)
            end
            
            #daggered rotations
            push!(circuit, chain(n_qubits, put(p=>chain(RotationGate(-1*get_pauli_circuit(i,p,circuit_spec,n_qubits),0))) for p = n_qubits:-1:1))

            counter-=n_qubits

        end
        #H squareroot, use matblock(mat()) to fix parameters. There may be better way to do this
        push!(circuit, chain(n_qubits, put(i=>InvSqrtH) for i = n_qubits:-1:1)) 

    end


    return circuit
end



function get_HC_from_pauli(weight,pauli_strings)
    ab=[]
    for i in 1:length(pauli_strings)
        push!(ab,weight[i]*get_pauli_op(pauli_strings[i]))
    end

    return sum(ab)

end




function calculate_circuit(qcbm,cost_op,circuit_spec,qcbm_daggered)
    n_qubits=nqubits(qcbm)

    function get_loss(qcbm,hc)
        reg=zero_state(n_qubits) |>qcbm
        loss= expect(hc, reg)|> real
        return loss
    end


    function get_gradient(qcbm,hc)
        ##do_gtensor_here is just additional switch to calculate QFIM in case qng==0
        reg=zero_state(n_qubits) => qcbm #whenver get gradients use =>
        g_reg,pure_gradient=expect'(hc, reg)
        return pure_gradient
    end

    function get_QFIM(qcbm,qcbm_daggered)

        n_parameters=nparameters(qcbm)
        Gtensor=zeros(n_parameters,n_parameters)


        expect_single=zeros(depth,n_qubits)
        #do diagonal entries
 
        for j in 1:depth
            sub_qcbm=qcbm[1:2*j-1]
            reg=zero_state(n_qubits) |> sub_qcbm #|> immediatley evaluate register, instead of => (is evaluated only when expect is calculated)

            for k in 1:n_qubits
                op_p=get_pauli_circuit(j,k,circuit_spec,n_qubits)
                op_y=1/2*put(n_qubits, k=>op_p)
                expct= expect(op_y, reg)|> real
                expect_single[j,k]=expct
            end

            expect_double=zeros(n_qubits,n_qubits)
            for k in 1:n_qubits
                for m in k:n_qubits
                    if k!=m #if same, resolves to identity for paulis
                        op_pk=get_pauli_circuit(j,k,circuit_spec,n_qubits)
                        op_pm=get_pauli_circuit(j,m,circuit_spec,n_qubits)
                        op_yy=1/4*chain(n_qubits, [put(k=>op_pk),put(m=>op_pm)] )
                    else
                        op_yy=1/4*igate(n_qubits)
                    end
                    expct= expect(op_yy, reg)|> real
                    expect_double[k,m]=expct
                end

            end

            #geometric tensor components needed are real hermitian
            for k in 1:n_qubits
                for m in k+1:n_qubits
                    expect_double[m,k]=expect_double[k,m]
                end
            end

            for k in 1:n_qubits
                for m in 1:n_qubits
                    Gtensor[(j-1)*n_qubits+k,(j-1)*n_qubits+m]=expect_double[k,m]-expect_single[j,k]*expect_single[j,m]
                end
            end


        end

        #gradients for depth n>j
        #build gradients for U_n=R_y(theta_n)*Cphase, U_0=R_y(pi/4)=qcbm[1]
        #<0|U_0^\dag U_1^\dag U_j^\dag U_2 (\partial_n U_n^\dag) (U_3^\dag U_3) U_n U_2 U_j (\partial_j U_j)U_1 U_0|0>
        for j in 1:depth
            sub_qcbm=qcbm[1:2*j-1]
            reg=zero_state(n_qubits) |> sub_qcbm #|> immediatley evaluate register, instead of => (is evaluated only when expect is calculated)
            for n in j+1:depth#min(j,depth)#depth#min(j+1,depth)


                right_qcbm=qcbm[2*j:2*n-1]#goes from depth j to beginning of depth n 
                
                #get sub circuit from daggered circuit. DO not use daggered function since it is horrible slow..
                if use_yao_adjoint==true
                    left_qcbm=Daggered(qcbm[2*j:2*n-1]) #goes from depth n to depth j as adjoint, Daggered is very slow!
                else
                    circuit_length=length(qcbm_daggered)
                    left_qcbm=qcbm_daggered[circuit_length+1-(2*n-1):circuit_length+1-2*j] #goes from depth n to depth j as adjoint
                end


                for k in 1:n_qubits
                    for m in 1:n_qubits
                        op_pjk=get_pauli_circuit(j,k,circuit_spec,n_qubits)
                        op_pnm=get_pauli_circuit(n,m,circuit_spec,n_qubits)
                        op_y_k=1/2*put(n_qubits, k=>op_pjk)#sits at qubit k at depth j
                        op_y_m=1/2*put(n_qubits, m=>op_pnm) # sits at qubit m at  depth n
                        op=left_qcbm*op_y_m*right_qcbm*op_y_k

                        expct= expect(op, reg)|> real #only real part needed for gtensor
                        Gtensor[(j-1)*n_qubits+k,(n-1)*n_qubits+m]=expct-expect_single[j,k]*expect_single[n,m]

                    end
                end
            end
        end
        for j in 1:depth
            for n in j+1:depth
                for k in 1:n_qubits
                    for m in 1:n_qubits
                        Gtensor[(n-1)*n_qubits+m,(j-1)*n_qubits+k]=Gtensor[(j-1)*n_qubits+k,(n-1)*n_qubits+m]
                    end
                end
            end
        end

        return Gtensor
    end






    loss=get_loss(qcbm,cost_op)
    gradient= get_gradient(qcbm,cost_op)
    QFIM= get_QFIM(qcbm,qcbm_daggered)


    return loss,gradient,QFIM
end


rng_param = MersenneTwister(rng_seed);
rng_circ= MersenneTwister(rng_seed+1);






if circuit_type==0 
    entangler_type=0 #cphase
elseif circuit_type==1 
    entangler_type=1 #CNOT
elseif circuit_type==2
    entangler_type=2#sqrt(iswap)
end

entanglingGates=[]
tempEnt=vcat([[2*i-1, ((2*i-1+1-1) %(n_qubits))+1,entangler_type] for i = 1:fld(n_qubits,2)],
[[2*i, ((2*i+1-1) %(n_qubits))+1,entangler_type] for i = 1:fld(n_qubits-1,2)]) 
entanglingGates=[tempEnt for p =1:depth]


#x,y,z rotations randomly
circuit_spec=[rand(rng_circ,1:3) for i in 1:depth*n_qubits] 


n_var_parameters=depth*n_qubits

if(choose_param_type==0)
    var_param=rand(rng_param,n_var_parameters)*2*pi #list of random parameters
elseif(choose_param_type==1)
    var_param=zeros(rng_param,n_var_parameters) #list of 0 parameters
end



hilbertspace=2^n_qubits

##get Hamiltonian. Is the ZZ pauli operator acting on first 2 qubits
hamilton_paulis=[]
n_terms=1
hc_pauli_weight=ones(n_terms)
hc_pauli_strings=[zeros(Int,n_qubits) for i in 1:n_terms]
hc_pauli_strings[1][1:2]=[3,3]

hamilton_paulis=hc_pauli_strings
cost_op=get_HC_from_pauli(hc_pauli_weight,hamilton_paulis)



println("Nparameters ",n_var_parameters)


qcbm = build_circuit(n_qubits, depth,entanglingGates,circuit_spec)
dispatch!(qcbm,var_param) # initialize the parameters for circuit
##construct inverse of circuit. This is needed for QFIM
qcbm_daggered=build_circuit(n_qubits, depth,entanglingGates,circuit_spec,true)#daggered circuit
dispatch!(qcbm_daggered, var_param[length(var_param):-1:1]) # initialize the parameters for daggered circuit


loss,gradient, QFIM = calculate_circuit(qcbm, cost_op,circuit_spec,qcbm_daggered)

println("Loss ",loss)

println("Gradient mean ",mean(gradient))
println("Gradient variance ",var(gradient))


##2D plot of the QFIM
gr()
data = QFIM
fig2D1=heatmap(1:size(data,1),
    1:size(data,2), data,clim=(-0.25,0.25),
    c=cgrad([:blue, :white,:red]),
    xlabel="params", ylabel="params")
display(fig2D1)


#get eigenvalues of the QFIM
eigenvalues=eigvals(QFIM)

count_threshold=0
e_val_threshold=[] #eigenvalues greater than threshold
for i in 1:length(eigenvalues)
    if eigenvalues[i]>=epsilon_inv
        push!(e_val_threshold,eigenvalues[i])
    end
end

println("Non-zero eigenvalues of QFIM ",length(e_val_threshold))

println("Total time ",time()-starttime)




