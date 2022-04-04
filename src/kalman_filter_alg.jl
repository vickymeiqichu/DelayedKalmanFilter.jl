# Base.@kwdef struct ModelParameters{T<:AbstractFloat}
#     P₀::T = 10000.0
#     h::T = 5.0
#     μₘ::T = log(2) / 30
#     μₚ::T = log(2) / 90
#     αₘ::T = 1.0
#     αₚ::T = 1.0
#     τ::T = 29.0
# end

# struct StateSpace{T<:AbstractFloat}
#     mean::Matrix{T}
#     variance::Matrix{T}
# end

# struct StateAndDistributions{T<:AbstractFloat}
#     state_space::StateSpace{T}
#     distributions::Vector{Normal{T}}
# end

struct TimeConstructor{T<:Integer}
    number_of_observations::T
    observation_time_step::T
    discretisation_time_step::T
    discrete_delay::T
    initial_number_of_states::T
    number_of_hidden_states::T
    total_number_of_states::T
end

"""
Function which returns an instance of TimeConstructor, which is defines the numbers of
states for various uses in the Kalman filter.
"""
function time_constructor_function(
    protein_at_observations::Matrix{<:AbstractFloat},
    τ::AbstractFloat,
    discretisation_time_step::Integer = 1,
)

    number_of_observations = size(protein_at_observations, 1)
    observation_time_step =
        Int(protein_at_observations[2, 1] - protein_at_observations[1, 1])

    discrete_delay = Int64(round(τ / discretisation_time_step))
    initial_number_of_states = discrete_delay + 1
    number_of_hidden_states = Int64(round(observation_time_step / discretisation_time_step))

    total_number_of_states =
        initial_number_of_states + (number_of_observations - 1) * number_of_hidden_states

    return TimeConstructor{Integer}(
        number_of_observations,
        observation_time_step,
        discretisation_time_step,
        discrete_delay,
        initial_number_of_states,
        number_of_hidden_states,
        total_number_of_states,
    )
end

"""
Calculate the mean and variance of the Normal approximation of the state space for a given
time point in the Kalman filtering algorithm and return it as a Normal distribution
"""
function distribution_prediction_at_given_time(
    state_space_mean,
    state_space_variance,
    states::TimeConstructor,
    given_time::Integer,
    observation_transform::Matrix{<:AbstractFloat},
    measurement_variance::AbstractFloat,
)

    mean_prediction = dot(observation_transform, state_space_mean[given_time, 2:3])
    last_predicted_covariance_matrix = state_space_variance[
        [given_time, states.total_number_of_states + given_time],
        [given_time, states.total_number_of_states + given_time],
    ]
    variance_prediction =
        dot(
            observation_transform,
            last_predicted_covariance_matrix * observation_transform',
        ) + measurement_variance

    return Normal(mean_prediction, sqrt(variance_prediction))
end

# if no time given, use initial number of states
function distribution_prediction_at_given_time(
    state_space_mean,
    state_space_variance,
    states::TimeConstructor,
    observation_transform::Matrix{<:AbstractFloat},
    measurement_variance::AbstractFloat,
)

    mean_prediction =
        dot(observation_transform, state_space_mean[states.initial_number_of_states, 2:3])
    last_predicted_covariance_matrix = state_space_variance[
        [
            states.initial_number_of_states,
            states.total_number_of_states + states.initial_number_of_states,
        ],
        [
            states.initial_number_of_states,
            states.total_number_of_states + states.initial_number_of_states,
        ],
    ]
    variance_prediction =
        dot(
            observation_transform,
            last_predicted_covariance_matrix * observation_transform',
        ) + measurement_variance

    return Normal(mean_prediction, sqrt(variance_prediction))
end

"""
Calculate the current number of states for a given observation time
"""
function calculate_current_number_of_states(
    current_observation_time::AbstractFloat,
    states::TimeConstructor,
    τ::AbstractFloat,
)
    return Int(round(current_observation_time / states.observation_time_step)) *
           states.number_of_hidden_states + states.initial_number_of_states
end

"""
Perform a delay-adjusted non-linear stochastic Kalman filter based on observation of protein
copy numbers. This implements the filter described by Calderazzo et al., Bioinformatics (2018).

# Arguments

- `protein_at_observations::AbstractArray{<:Real}`: Observed protein. The dimension is n x 2, where n is the number of observation time points.
    The first column is the time, and the second column is the observed protein copy number at
    that time. The filter assumes that observations are generated with a fixed, regular time interval.

- `model_parameters::ModelParameters`: A ModelParameters object containing the model parameters in the following order:
    repression threshold, hill coefficient, mRNA degradation rate,protein degradation rate, basal transcription rate,
    translation rate, time delay.

- `measurement_variance::Real`: The variance in our measurement. This is given by Sigma epsilon in Calderazzo et. al. (2018).

# Returns

- `state_space_and_distributions::StateAndDistributions`: TODO
"""
function kalman_filter(
    protein_at_observations::Matrix{<:AbstractFloat},
    model_parameters::Vector{<:AbstractFloat},
    measurement_variance::AbstractFloat,
)

    τ = model_parameters[7]
    # F in the paper
    observation_transform = [0.0 1.0]
    states = time_constructor_function(protein_at_observations, τ)
    # initialise state space and distribution predictions
    state_space_mean, state_space_variance, predicted_observation_distributions =
    kalman_filter_state_space_initialisation(
        protein_at_observations,
        model_parameters,
        states,
        observation_transform,
        measurement_variance,
    )

    # loop through observations and at each observation apply the Kalman prediction step and then the update step
    for observation_index = 2:size(protein_at_observations, 1)
        current_observation = protein_at_observations[observation_index, :]
        kalman_prediction_step!(
            state_space_mean,
            state_space_variance,
            states,
            current_observation,
            model_parameters,
        )
        current_number_of_states = calculate_current_number_of_states(
            current_observation[1],
            states,
            τ
        )
        # between the prediction and update steps we record the normal distributions for our likelihood
        predicted_observation_distributions[observation_index] =
            distribution_prediction_at_given_time(
                state_space_mean,
                state_space_variance,
                states,
                current_number_of_states,
                observation_transform,
                measurement_variance,
            )
        kalman_update_step!(
            state_space_mean,
            state_space_variance,
            states,
            current_observation,
            τ,
            measurement_variance,
            observation_transform,
        )
    end # for
    return state_space_mean, state_space_variance, predicted_observation_distributions
end # function

"""
Initialse the state space mean for a given set of time states and the ODE system's steady state
"""
function initialise_state_space_mean(states::TimeConstructor, steady_state)

    state_space_mean = Matrix{Float64}(undef, states.total_number_of_states, 3)

    state_space_mean[:, 1] = LinRange(
        -states.discrete_delay,
        states.total_number_of_states - states.discrete_delay - 1,
        states.total_number_of_states,
    )
    state_space_mean[1:states.initial_number_of_states, 2] .= steady_state[1]
    state_space_mean[1:states.initial_number_of_states, 3] .= steady_state[2]

    return state_space_mean
end

"""
Initialse the state space variance for a given set of time states and the ODE system's steady state
"""
function initialise_state_space_variance(
    states::TimeConstructor,
    steady_state;
    mRNA_scaling::AbstractFloat = 20.0,
    protein_scaling::AbstractFloat = 100.0,
)

    state_space_variance = Matrix{Float64}(
        undef,
        2*states.total_number_of_states,
        2*states.total_number_of_states,
    )

    diag_indices = diagind(state_space_variance)
    mRNA_indices = 1:states.initial_number_of_states
    protein_indices =
        1+states.total_number_of_states:1+states.total_number_of_states+states.initial_number_of_states

    state_space_variance[diag_indices[mRNA_indices]] .= steady_state[1] * mRNA_scaling
    state_space_variance[diag_indices[protein_indices]] .= steady_state[2] * protein_scaling

    return state_space_variance
end

"""
A function for initialisation of the state space mean and variance, and update for the "negative" times that
are a result of the time delay. Initialises the negative times using the steady state of the deterministic system,
and then updates them with kalman_update_step.

# Arguments

- `protein_at_observations::AbstractArray{<:Real}`: Observed protein. The dimension is n x 2, where n is the number of observation time points.
    The first column is the time, and the second column is the observed protein copy number at
    that time. The filter assumes that observations are generated with a fixed, regular time interval.

- `model_parameters::ModelParameters`: A ModelParameters object containing the model parameters in the following order:
    repression threshold, hill coefficient, mRNA degradation rate,protein degradation rate, basal transcription rate,
    translation rate, time delay.

- `measurement_variance::Real`: The variance in our measurement. This is given by Sigma epsilon in Calderazzo et. al. (2018).

# Returns

- `state_and_distributions::StateAndDistributions`: TODO
"""
function kalman_filter_state_space_initialisation(
    protein_at_observations::Matrix{<:AbstractFloat},
    model_parameters::Vector{<:AbstractFloat},
    states::TimeConstructor,
    observation_transform,
    measurement_variance::AbstractFloat = 10.0,
)

    τ = model_parameters[7]

    steady_state = calculate_steady_state_of_ode(model_parameters)

    # construct state space
    state_space_mean = initialise_state_space_mean(states, steady_state)
    state_space_variance = initialise_state_space_variance(states, steady_state)
    # state_space = StateSpace{Float64}(state_space_mean, state_space_variance)

    # initialise distributions
    predicted_observation_distributions =
        Array{Normal{Float64}}(undef, states.number_of_observations)
    predicted_observation_distributions[1] = distribution_prediction_at_given_time(
        state_space_mean,
        state_space_variance,
        states,
        observation_transform,
        measurement_variance,
    )
    # update the past ("negative time")
    current_observation = protein_at_observations[1, :]
    kalman_update_step!(
        state_space_mean,
        state_space_variance,
        states,
        current_observation,
        τ,
        measurement_variance,
        observation_transform,
    )

    return state_space_mean, state_space_variance, predicted_observation_distributions
end # function

function construct_instant_jacobian(model_parameters)
    [
        -model_parameters[3] 0.0
        model_parameters[6] -model_parameters[4]
    ]
end

function construct_delayed_jacobian(model_parameters, past_protein)
    [
        0.0 model_parameters[5]*∂hill∂p(past_protein, model_parameters[1], model_parameters[2])
        0.0 0.0
    ]
end

"""
Returns past time, current mean and past protein
"""
function get_current_and_past_mean(state_space_mean, current_time_index, discrete_delay)
    return current_time_index - discrete_delay,
    state_space_mean[current_time_index, 2:3],
    state_space_mean[current_time_index-discrete_delay, 3]
end

"""
Predict state space mean to the next observation time index
"""
function predict_state_space_mean!(
    state_space_mean,
    current_number_of_states,
    current_mean,
    model_parameters,
    states
    )

    function state_space_mean_RHS(du,u,p,t) # is there some way for this function to not be nested? does it matter?
        # past_protein = h(p,t-p[7]) # TODO define history function
        past_index = Int(t) - states.discrete_delay
        past_protein = state_space_mean[past_index,3] # currently indexing array rather than using history function
        du[1] = -p[3]*u[1] + p[5]*hill_function(past_protein, p[1], p[2])
        du[2] = p[6]*u[1] - p[4]*u[2]
    end

    tspan = (current_number_of_states,current_number_of_states+states.number_of_hidden_states)
    mean_prob = ODEProblem(state_space_mean_RHS,
                           current_mean,#state_space_mean[current_number_of_states,[2,3]],
                           # mean_history,
                           tspan,
                           model_parameters)#; constant_lags=[transcription_delay])
    mean_solution = solve(mean_prob,Euler(),dt=1.,adaptive=false,saveat=1.,mindt=1.,maxdt=1.)
    # mean_solution_dict[prediction_index] = mean_solution
    state_space_mean[(current_number_of_states + 1):(current_number_of_states + states.number_of_hidden_states),2:3] = hcat(mean_solution.(tspan[1]+1:tspan[2])...)'#mean_solution.(tspan[1]+1:tspan[2])
    # state_space_mean[(current_number_of_states + 1):(current_number_of_states + states.number_of_hidden_states),2] = [mean_solution(i)[1] for i in tspan[1]+1:tspan[2]]
    # state_space_mean[(current_number_of_states + 1):(current_number_of_states + states.number_of_hidden_states),3] = [mean_solution(i)[2] for i in tspan[1]+1:tspan[2]]
end

"""
Predict state space variance to the next observation time index
"""
function predict_state_space_variance!(
    state_space_mean,
    state_space_variance,
    current_number_of_states,
    model_parameters,
    states,
    instant_jacobian
    )

    function state_space_variance_RHS(du,u,p,t)
        # past_protein = h(p,t-p[7])
        past_index = Int(t) - states.discrete_delay
        # assign diagonal prediction so we can do the off diagonals at each step
        state_space_variance[[Int(t),states.total_number_of_states+Int(t)],
                             [Int(t),states.total_number_of_states+Int(t)]] = u

        past_protein = state_space_mean[past_index,3]
        current_mean = state_space_mean[Int(t),2:3]
        past_to_now_diagonal_variance = state_space_variance[[past_index,states.total_number_of_states+past_index],
                                                             [Int(t),states.total_number_of_states+Int(t)]]
        delayed_jacobian = construct_delayed_jacobian(model_parameters, past_protein)

        variance_of_noise = [p[3]*current_mean[1]+p[5]*hill_function(past_protein, p[1], p[2]) 0.0;
                             0.0 p[6]*current_mean[1]+p[4]*current_mean[2]]

        du .= instant_jacobian*u + u*instant_jacobian' +
             delayed_jacobian*past_to_now_diagonal_variance +
             past_to_now_diagonal_variance'*delayed_jacobian' +
             variance_of_noise

        ## at each step we need to calculate off diagonals
        if t < tspan[2]
            for intermediate_time_index in past_index:Int(t)

                function off_diagonal_state_space_variance_RHS(du,u,p,diag_t)
                    covariance_matrix_intermediate_to_past = state_space_variance[[Int(diag_t),
                                                                                   states.total_number_of_states+Int(diag_t)],
                                                                                  [past_index,
                                                                                   states.total_number_of_states+past_index]]

                    du .= u*instant_jacobian' +
                         covariance_matrix_intermediate_to_past*delayed_jacobian'
                end # function

                covariance_matrix_intermediate_to_current = state_space_variance[[intermediate_time_index,
                                                                                  states.total_number_of_states+intermediate_time_index],
                                                                                  [Int(t),
                                                                                   states.total_number_of_states+Int(t)]]

                diag_tspan = (Float64(intermediate_time_index),Float64(intermediate_time_index)+1)
                off_diag_prob = ODEProblem(off_diagonal_state_space_variance_RHS,
                                           covariance_matrix_intermediate_to_current,# P(s,t)
                                           # mean_history,
                                           diag_tspan,
                                           model_parameters)#; constant_lags=[transcription_delay])
                off_diag_solution = solve(off_diag_prob,Euler(),dt=1.,adaptive=false,saveat=1.,mindt=1.,maxdt=1.)

                # Fill in the big matrix
                state_space_variance[[intermediate_time_index,
                                      states.total_number_of_states+intermediate_time_index],
                                     [Int(t)+1,
                                      states.total_number_of_states+Int(t)+1]] = off_diag_solution(intermediate_time_index+1)
                state_space_variance[[Int(t)+1,
                                      states.total_number_of_states+Int(t)+1],
                                     [intermediate_time_index,
                                      states.total_number_of_states+intermediate_time_index]] = off_diag_solution(intermediate_time_index+1)'
            end # intermediate time index for
        end
    end
    tspan = (current_number_of_states,current_number_of_states+states.number_of_hidden_states)
    current_variance = state_space_variance[[current_number_of_states, states.total_number_of_states + current_number_of_states],
                                            [current_number_of_states, states.total_number_of_states + current_number_of_states]]
    variance_prob = ODEProblem(state_space_variance_RHS,
                               current_variance,
                               tspan,
                               model_parameters)
    solve(variance_prob,Euler(),dt=1.,adaptive=false,saveat=1.,mindt=1.,maxdt=1.)
end

"""
Predict state space mean one time step forward to 'next_time_index'
"""
function predict_state_space_mean_one_step!(
    state_space_mean,
    current_time_index,
    current_mean,
    model_parameters,
    states,
    hill_function_value,
)
    # derivative of mean is contributions from instant reactions + contributions from past reactions
    derivative_of_mean = (
        [
            -model_parameters[3] 0.0;
            model_parameters[6] -model_parameters[4]
        ] * current_mean .+ [model_parameters[5] * hill_function_value, 0.0]
    )

    next_mean = current_mean .+ states.discretisation_time_step*derivative_of_mean
    # ensures the prediction is non negative
    next_mean = max.(next_mean,0.0)
    state_space_mean[current_time_index+1, 2:3] = next_mean

    return state_space_mean
end

"""
Calculate the derivative of the variance needed for the forward Euler integrator
"""
function calculate_variance_derivative(
    state_space_variance,
    current_time_index,
    current_mean,
    model_parameters,
    states,
    hill_function_value,
    instant_jacobian,
    delayed_jacobian,
)

    past_time_index = current_time_index - states.discrete_delay

    # P(t,t)
    @views current_covariance_matrix = state_space_variance[
        [current_time_index, states.total_number_of_states + current_time_index],
        [current_time_index, states.total_number_of_states + current_time_index],
    ]

    # P(t-tau,t)
    @views covariance_matrix_past_to_now = state_space_variance[
        [past_time_index, states.total_number_of_states + past_time_index],
        [current_time_index, states.total_number_of_states + current_time_index],
    ]

    # P(t,t-tau)
    @views covariance_matrix_now_to_past = state_space_variance[
        [current_time_index, states.total_number_of_states + current_time_index],
        [past_time_index, states.total_number_of_states + past_time_index],
    ]

    variance_change_current_contribution = instant_jacobian * current_covariance_matrix +
        current_covariance_matrix * instant_jacobian'

    variance_change_past_contribution = delayed_jacobian * covariance_matrix_past_to_now +
        covariance_matrix_now_to_past * delayed_jacobian'

    variance_of_noise = [
        model_parameters[3]*current_mean[1]+model_parameters[5]*hill_function_value 0.0
        0.0 model_parameters[6]*current_mean[1]+model_parameters[4]*current_mean[2]
    ]

    derivative_of_variance = variance_change_current_contribution +
        variance_change_past_contribution +
        variance_of_noise

    return derivative_of_variance, current_covariance_matrix

end

function predict_diagonal_variance_one_step!(
    state_space_variance,
    current_time_index,
    current_mean,
    model_parameters,
    states,
    hill_function_value,
    instant_jacobian,
    delayed_jacobian,
)

    derivative_of_variance, current_covariance_matrix = calculate_variance_derivative(
        state_space_variance,
        current_time_index,
        current_mean,
        model_parameters,
        states,
        hill_function_value,
        instant_jacobian,
        delayed_jacobian,
    )

    # P(t+Deltat,t+Deltat)
    next_covariance_matrix =
        current_covariance_matrix +
        states.discretisation_time_step*derivative_of_variance
    # ensure that the diagonal entries are non negative
    idx = diagind(next_covariance_matrix)

    next_covariance_matrix[idx] = max.(next_covariance_matrix[idx], 0.0)


    state_space_variance[
        [current_time_index + 1, states.total_number_of_states + current_time_index + 1],
        [current_time_index + 1, states.total_number_of_states + current_time_index + 1],
    ] = next_covariance_matrix

    return state_space_variance
end

"""
Calculate the derivative of the covariance needed for the forward Euler intergator
"""
function calculate_covariance_derivative(
    state_space_variance,
    current_time_index,
    intermediate_time_index,
    current_mean,
    states,
    instant_jacobian,
    delayed_jacobian,
)

    past_time_index = current_time_index - states.discrete_delay

    # P(s,t)
    @views covariance_matrix_intermediate_to_current = state_space_variance[
        [intermediate_time_index, states.total_number_of_states + intermediate_time_index],
        [current_time_index, states.total_number_of_states + current_time_index],
    ]
    # P(s,t-tau)
    @views covariance_matrix_intermediate_to_past = state_space_variance[
        [intermediate_time_index, states.total_number_of_states + intermediate_time_index],
        [past_time_index, states.total_number_of_states + past_time_index],
    ]


    return covariance_matrix_intermediate_to_current * instant_jacobian' +
        covariance_matrix_intermediate_to_past * delayed_jacobian',
        covariance_matrix_intermediate_to_current
end

function predict_off_diagonal_variance_one_step!(
    state_space_variance,
    current_time_index,
    intermediate_time_index,
    current_mean,
    states,
    instant_jacobian,
    delayed_jacobian,
)

    covariance_derivative, covariance_matrix_intermediate_to_current =
        calculate_covariance_derivative(
            state_space_variance,
            current_time_index,
            intermediate_time_index,
            current_mean,
            states,
            instant_jacobian,
            delayed_jacobian,
        )

    # This corresponds to P(s,t+Deltat) in the Calderazzo paper
    covariance_matrix_intermediate_to_next =
        covariance_matrix_intermediate_to_current +
        states.discretisation_time_step * covariance_derivative

    # Fill in the big matrix
    state_space_variance[
        [intermediate_time_index, states.total_number_of_states + intermediate_time_index],
        [current_time_index + 1, states.total_number_of_states + current_time_index + 1],
    ] = covariance_matrix_intermediate_to_next
    # Fill in the big matrix with transpose arguments, i.e. P(t+Deltat, s) - works if initialised symmetrically
    state_space_variance[
        [current_time_index + 1, states.total_number_of_states + current_time_index + 1],
        [intermediate_time_index, states.total_number_of_states + intermediate_time_index],
    ] = covariance_matrix_intermediate_to_next'

    return state_space_variance
end

"""
Perform the Kalman filter prediction about future observation, based on current knowledge i.e. current
state space mean and variance. This gives rho_{t+delta t-tau:t+delta t} and P_{t+delta t-tau:t+delta t},
using the differential equations in supplementary section 4 of Calderazzo et al., Bioinformatics (2018),
approximated using a forward Euler scheme.

# Arguments

- `state_space::StateSpace`: TODO

- `states::TimeConstructor`: TODO

- `current_observation::Vector{<:AbstractFloat}`: TODO

- `model_parameters::ModelParameters`: A ModelParameters object containing the model parameters in the following order:
    repression threshold, hill coefficient, mRNA degradation rate,protein degradation rate, basal transcription rate,
    translation rate, time delay.

# Returns

- `state_space::StateSpace`: TODO
"""
function kalman_prediction_step!(
    state_space_mean,
    state_space_variance,
    states::TimeConstructor,
    current_observation::Vector{<:AbstractFloat},
    model_parameters::Vector{<:AbstractFloat},
)

    τ = model_parameters[7]

    # @unpack P₀, h, μₘ, μₚ, αₘ, αₚ, τ = model_parameters
    # this is the number of states at t, i.e. before predicting towards t+observation_time_step
    current_number_of_states = calculate_current_number_of_states(
        current_observation[1] - states.number_of_hidden_states,
        states,
        τ,
    )

    instant_jacobian = construct_instant_jacobian(model_parameters)

    predict_state_space_mean!(
        state_space_mean,
        current_number_of_states,
        state_space_mean[current_number_of_states, 2:3],
        model_parameters,
        states
        )

    predict_state_space_variance!(
        state_space_mean,
        state_space_variance,
        current_number_of_states,
        model_parameters,
        states,
        instant_jacobian
        )

    # for current_time_index =
    #     current_number_of_states:current_number_of_states+states.number_of_hidden_states-1
    #
    #     past_time_index, current_mean, past_protein = get_current_and_past_mean(
    #         state_space_mean,
    #         current_time_index,
    #         states.discrete_delay,
    #     )
    #
    #     # delayed_jacobian derivative of f with respect to past state ([past_mRNA, past_protein])
    #     delayed_jacobian = construct_delayed_jacobian(model_parameters, past_protein)
    #
    #     # predict_state_space_mean_one_step!(
    #     #     state_space_mean,
    #     #     current_time_index,
    #     #     current_mean,
    #     #     model_parameters,
    #     #     states,
    #     #     hill_function(past_protein, model_parameters[1], model_parameters[2]),
    #     # )
    #
    #     predict_diagonal_variance_one_step!(
    #         state_space_variance,
    #         current_time_index,
    #         current_mean,
    #         model_parameters,
    #         states,
    #         hill_function(past_protein, model_parameters[1], model_parameters[2]),
    #         instant_jacobian,
    #         delayed_jacobian,
    #     )
    #
    #     # predict the cross correlations, P(t-τ:t,t+Δt)
    #     for intermediate_time_index = past_time_index:current_time_index
    #         predict_off_diagonal_variance_one_step!(
    #             state_space_variance,
    #             current_time_index,
    #             intermediate_time_index,
    #             current_mean,
    #             states,
    #             instant_jacobian,
    #             delayed_jacobian,
    #         )
    #     end
    # end # for
    return state_space_mean, state_space_variance
end # function

"""
Take the stacked state space mean and return the updated state space mean in the original dimensions
"""
function update_stacked_state_space_mean(
    stacked_state_space_mean,
    adaptation_coefficient,
    observation_transform,
    current_observation,
    predicted_final_state_space_mean,
    states,
)

    stacked_state_space_mean .+= reshape(
        adaptation_coefficient * (
            current_observation[2] -
            dot(observation_transform, predicted_final_state_space_mean)
        ),
        length(stacked_state_space_mean),
    )
    # ensure the mean mRNA and protein are non negative
    stacked_state_space_mean[stacked_state_space_mean.<0.0] .= 0.0

    updated_state_space_mean = hcat(
        stacked_state_space_mean[1:(states.discrete_delay+1)],
        stacked_state_space_mean[(states.discrete_delay+2):end],
    )

    return updated_state_space_mean
end

function update_mean!(
    state_space_mean,
    current_number_of_states,
    states,
    adaptation_coefficient,
    current_observation,
    observation_transform,
)

    # we update back to t-τ, so only consider these points
    shortened_state_space_mean = state_space_mean[
        (current_number_of_states-states.discrete_delay):current_number_of_states,
        [2, 3],
    ]

    # put protein values underneath mRNA values, to make vector of means
    # consistent with variance
    stacked_state_space_mean =
        vcat(shortened_state_space_mean[:, 1], shortened_state_space_mean[:, 2])

    predicted_final_state_space_mean = state_space_mean[current_number_of_states, [2, 3]]

    state_space_mean[
        (current_number_of_states-states.discrete_delay):current_number_of_states,
        [2, 3],
    ] .= update_stacked_state_space_mean(
        stacked_state_space_mean,
        adaptation_coefficient,
        observation_transform,
        current_observation,
        predicted_final_state_space_mean,
        states,
    )

    return state_space_mean
end

function calculate_helper_inverse(
    observation_transform,
    predicted_final_covariance_matrix,
    measurement_variance,
)

    1.0 / (
        dot(
            observation_transform,
            predicted_final_covariance_matrix * observation_transform',
        ) + measurement_variance
    )

end

function calculate_adaptation_coefficient(cov_matrix, observation_transform, helper_inverse)
    sum(dot.(cov_matrix, observation_transform), dims = 2) * helper_inverse
end

function calculate_shortened_covariance_matrix(
    state_space_variance,
    current_number_of_states,
    states::TimeConstructor,
)

    mRNA_indices_to_keep =
        (current_number_of_states-states.discrete_delay):(current_number_of_states)
    protein_indices_to_keep =
        (states.total_number_of_states+current_number_of_states-states.discrete_delay):(states.total_number_of_states+current_number_of_states)
    all_indices_up_to_delay = vcat(mRNA_indices_to_keep, protein_indices_to_keep)

    return state_space_variance[all_indices_up_to_delay, all_indices_up_to_delay],
    all_indices_up_to_delay
end

"""
Return the covariance matrix at the current (last predicted) number of states
"""
function calculate_final_covariance_matrix(
    state_space_variance,
    current_number_of_states,
    states,
)
    return state_space_variance[
        [
            current_number_of_states,
            states.total_number_of_states + current_number_of_states,
        ],
        [
            current_number_of_states,
            states.total_number_of_states + current_number_of_states,
        ],
    ]
end

function update_variance!(
    state_space_variance,
    shortened_covariance_matrix,
    all_indices_up_to_delay,
    states,
    adaptation_coefficient,
    observation_transform,
)

    updated_shortened_covariance_matrix = (
        shortened_covariance_matrix -
        adaptation_coefficient *
        observation_transform *
        shortened_covariance_matrix[[states.discrete_delay + 1, end], :]
    )

    idx = diagind(updated_shortened_covariance_matrix)
    updated_shortened_covariance_matrix[idx] = max.(updated_shortened_covariance_matrix[idx], 0.0)

    # Fill in updated values
    state_space_variance[all_indices_up_to_delay, all_indices_up_to_delay] =
        updated_shortened_covariance_matrix

    return state_space_variance
end

"""
Perform the Kalman filter update step on the predicted mean and variance, given a new observation.
This implements the equations at the beginning of page 4 in Calderazzo et al., Bioinformatics (2018).
This assumes that the observations are collected at fixed time intervals.

# Arguments

- `state_space::StateSpace`: TODO

- `states::TimeConstructor`: TODO

- `current_observation::AbstractArray{<:Real}`: TODO

- `τ::Real`: TODO

- `measurement_variance::Real`: TODO

# Returns

- `state_space::StateSpace`: TODO

"""
function kalman_update_step!(
    state_space_mean,
    state_space_variance,
    states::TimeConstructor,
    current_observation::Vector{<:AbstractFloat},
    τ::AbstractFloat,
    measurement_variance::AbstractFloat,
    observation_transform,
)

    current_number_of_states =
        calculate_current_number_of_states(current_observation[1], states, τ)

    # extract covariance matrix up to delay and indices
    # corresponds to P(t+Deltat-delay:t+deltat,t+Deltat-delay:t+deltat)
    shortened_covariance_matrix, all_indices_up_to_delay =
        calculate_shortened_covariance_matrix(
            state_space_variance,
            current_number_of_states,
            states,
        )

    # This is P(t+Deltat,t+Deltat) in the paper
    predicted_final_covariance_matrix = calculate_final_covariance_matrix(
        state_space_variance,
        current_number_of_states,
        states,
    )

    # This is (FP_{t+Deltat}F^T + Sigma_e)^-1
    helper_inverse = calculate_helper_inverse(
        observation_transform,
        predicted_final_covariance_matrix,
        measurement_variance,
    )

    # This is C in the paper
    adaptation_coefficient = calculate_adaptation_coefficient(
        shortened_covariance_matrix[:, [states.discrete_delay + 1, end]],
        observation_transform,
        helper_inverse,
    )
    # this is ρ*
    update_mean!(
        state_space_mean,
        current_number_of_states,
        states,
        adaptation_coefficient,
        current_observation,
        observation_transform,
    )
    # This is P*
    update_variance!(
        state_space_variance,
        shortened_covariance_matrix,
        all_indices_up_to_delay,
        states,
        adaptation_coefficient,
        observation_transform,
    )
    return state_space_mean, state_space_variance
end # function
