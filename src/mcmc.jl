"""
todo
"""
function kalman_filter_mh(protein_observations,measurement_variance,number_of_samples; full_model=false,n_threads=1)
    # create a single argument function
    f_test = x -> calculate_log_likelihood_at_parameter_point(protein_observations,x,measurement_variance)

   if full_model
      @model function full_kalman_model(protein_observations,::Type{T} = Float64) where {T}
          model_parameters = Vector{T}(undef,7)

          model_parameters[1] ~ Uniform(5000,15000) #repression_threshold
          model_parameters[2] ~ Uniform(2,6) #hill_coefficient
          model_parameters[3] ~ Uniform(log(2)/200,log(2)/10) #mRNA_degradation_rate
          model_parameters[4] ~ Uniform(log(2)/200,log(2)/10) #protein_degradation_rate
          model_parameters[5] ~ Uniform(0.01,120) #basal_transcription_rate
          model_parameters[6] ~ Uniform(0.01,40) #translation_rate
          model_parameters[7] ~ Uniform(1,40) #transcriptional_delay
          Turing.@addlogprob! f_test(model_parameters)
     end

      if n_threads > 1
          c1 = sample(full_kalman_model(protein_observations),MH(diagm(fill(1,7))),MCMCThreads(),number_of_samples,n_threads);
      else
          c1 = sample(full_kalman_model(protein_observations),MH(diagm(fill(1,7))),number_of_samples);
      end
      return c1
   end #if

    @model function kalman_model(protein_observations,::Type{T} = Float64) where {T}
          model_parameters = Vector{T}(undef,7)
          model_parameters[3] = log(2)/30
          model_parameters[4] = log(2)/90

          model_parameters[1] ~ Uniform(5000,15000) #repression_threshold
          model_parameters[2] ~ Uniform(2,6) #hill_coefficient
          model_parameters[5] ~ Uniform(0.01,120) #basal_transcription_rate
          model_parameters[6] ~ Uniform(0.01,40) #translation_rate
          model_parameters[7] ~ Uniform(1,40) #transcriptional_delay
          Turing.@addlogprob! f_test(model_parameters)
       end

       if n_threads > 1
           c1 = sample(kalman_model(protein_observations),MH(diagm(fill(1,5))),MCMCThreads(),number_of_samples,n_threads);
       else
           c1 = sample(kalman_model(protein_observations),MH(diagm(fill(1,5))),number_of_samples);
       end
    return c1
end #function

"""
todo
"""
function kalman_filter_NUTS(protein_observations,measurement_variance,number_of_samples; full_model=false,n_threads=1)
    # create single argument function
    f_test = x -> calculate_log_likelihood_at_parameter_point(protein_observations,x,measurement_variance)

   if full_model
      @model function full_kalman_model(protein_observations,::Type{T} = Float64) where {T}
          model_parameters = Vector{T}(undef,7)
          model_parameters[1] ~ Uniform(5000,15000) #repression_threshold
          model_parameters[2] ~ Uniform(2,6) #hill_coefficient
          model_parameters[3] ~ Uniform(log(2)/200,log(2)/10) #mRNA_degradation_rate
          model_parameters[4] ~ Uniform(log(2)/200,log(2)/10) #protein_degradation_rate
          model_parameters[5] ~ Uniform(0.01,120) #basal_transcription_rate
          model_parameters[6] ~ Uniform(0.01,40) #translation_rate
          model_parameters[7] ~ Uniform(1,40) #transcriptional_delay
          Turing.@addlogprob! f_test(model_parameters)
      end

      if n_threads > 1
          c1 = sample(full_kalman_model(protein_observations),DynamicNUTS(),MCMCThreads(),number_of_samples,n_threads);
      else
          c1 = sample(full_kalman_model(protein_observations),DynamicNUTS(),number_of_samples);
      end # if-else

      return c1
   end #if

   @model function kalman_model(protein_observations,::Type{T} = Float64) where {T}
       model_parameters = Vector{T}(undef,7)
       model_parameters[3] = log(2)/30
       model_parameters[4] = log(2)/90

       model_parameters[1] ~ Uniform(5000,15000) #repression_threshold
       model_parameters[2] ~ Uniform(2,6) #hill_coefficient
       # model_parameters[3] ~ Uniform(log(2)/200,log(2)/10) #mRNA_degradation_rate
       # model_parameters[4] ~ Uniform(log(2)/200,log(2)/10) #protein_degradation_rate
       model_parameters[5] ~ Uniform(0.01,120) #basal_transcription_rate
       model_parameters[6] ~ Uniform(0.01,40) #translation_rate
       model_parameters[7] ~ Uniform(1,40) #transcriptional_delay
       Turing.@addlogprob! f_test(model_parameters)
   end

   if n_threads > 1
        c1 = sample(kalman_model(protein_observations),NUTS(0.65),MCMCThreads(),number_of_samples,n_threads);
    else
        c1 = sample(kalman_model(protein_observations),NUTS(0.65),number_of_samples);
    end # if-else

   return c1
end #function


# """
# todo
# """
# function kalman_filter_smMALA(protein_at_observations,
#                               model_parameters,
#                               measurement_variance,
#                               number_of_samples,
#                               initial_position,
#                               step_size; thinning_rate=1,regularization_constant=1e+6)
#
#     # initialise the covariance proposal matrix
#     number_of_parameters = length(initial_position)
#
#     # initialise samples matrix and acceptance ratio counter
#     accepted_moves = 0.
#     mcmc_samples = zeros(number_of_samples,number_of_parameters)
#     mcmc_samples[1,:] .= initial_position
#     number_of_iterations = number_of_samples*thinning_rate
#
#     # initialise markov chain
#     current_position = initial_position
#     current_log_likelihood = calculate_log_likelihood_at_parameter_point(protein_at_observations,current_position,measurement_variance)
#     current_log_likelihood_gradient = calculate_log_likelihood_derivative_at_parameter_point(protein_at_observations,current_position,measurement_variance)
#     # we use the negative hessian of the positive log target
#     # and then regularize using the softabs metric, see Betancourt (2013)
#     current_log_likelihood_hessian = -calculate_log_likelihood_hessian_at_parameter_point(protein_at_observations,current_position,measurement_variance)
#     current_hessian_eigvals, current_hessian_eigvectors = eigen(current_log_likelihood_hessian)
#     current_regularized_eigvals = current_hessian_eigvals.*(1/tanh.(regularization_constant.*current_hessian_eigvals))
#     current_sqrt_inverse_softabs_hessian = current_hessian_eigvectors*diagm(1 ./ (sqrt.(current_regularized_eigvals)))
#     current_inverse_softabs_hessian = current_sqrt_inverse_softabs_hessian*transpose(current_sqrt_inverse_softabs_hessian)
#     current_softabs_hessian = current_hessian_eigvectors*diagm(current_regularized_eigvals)*transpose(current_hessian_eigvectors)
#
#     for iteration_index in 2:number_of_iterations
#         # progress measure
#         if iteration_index%(number_of_iterations/10)==0
#             println(string("Progress: ",100*iteration_index//number_of_iterations,'%'))
#         end
#
#         proposal = current_position .+ step_size.*(current_inverse_softabs_hessian*(current_log_likelihood_gradient./2)) +
#                                            sqrt(step_size).*(current_sqrt_inverse_softabs_hessian*rand(Normal(),number_of_parameters))
#         if proposal_log_likelihood == -Inf
#            if iteration_index%thinning_rate == 0
#                mcmc_samples[Int64(iteration_index/thinning_rate),:] .= current_position
#                continue
#            end
#         end
#         proposal_log_likelihood = calculate_log_likelihood_at_parameter_point(protein_at_observations,proposal,measurement_variance)
#         proposal_log_likelihood_gradient = calculate_log_likelihood_deriviative_at_parameter_point(protein_at_observations,proposal,measurement_variance)
#         # we use the negative hessian of the positive log target
#         # and then regularize using the softabs metric, see Betancourt (2013)
#         proposal_log_likelihood_hessian = -calculate_log_likelihood_hessian_at_parameter_point(protein_at_observations,proposal,measurement_variance)
#         proposal_hessian_eigvals, proposal_hessian_eigvectors = eigen(proposal_log_likelihood_hessian)
#         proposal_regularized_eigvals = proposal_hessian_eigvals.*(1/tanh.(regularization_constant.*proposal_hessian_eigvals))
#         proposal_sqrt_inverse_softabs_hessian = proposal_hessian_eigvectors*diagm(1 ./ (sqrt.(proposal_regularized_eigvals)))
#         proposal_inverse_softabs_hessian = proposal_sqrt_inverse_softabs_hessian*transpose(proposal_sqrt_inverse_softabs_hessian)
#         proposal_softabs_hessian = proposal_hessian_eigvectors*diagm(proposal_regularized_eigvals)*transpose(proposal_hessian_eigvectors)
#
#         forward_helper_variable = proposal .- current_position .- step_size.*(current_inverse_softabs_hessian*(current_log_likelihood_gradient./2))
#         backward_helper_variable = current_position .- proposal .- step_size.*(proposal_inverse_softabs_hessian*(proposal_log_likelihood_gradient./2))
#
#         transition_kernel_pdf_forward = 0.5*sum(log.(current_regularized_eigvals))-(transpose(forward_helper_variable)*current_softabs_hessian*forward_helper_variable)/(2*step_size)
#         transition_kernel_pdf_backward = 0.5*sum(log.(proposal_regularized_eigvals))-(transpose(backward_helper_variable)*proposal_softabs_hessian*backward_helper_variable)/(2*step_size)
#
#         if rand() < exp(proposal_log_likelihood - transition_kernel_pdf_forward - current_log_likelihood + transition_kernel_pdf_backward)
#             current_position .= proposal
#             current_log_likelihood = proposal_log_likelihood
#             current_log_likelihood_gradient .= proposal_log_likelihood_gradient
#             current_regularized_eigvals .= proposal_regularized_eigvals
#             current_sqrt_inverse_softabs_hessian .= proposal_sqrt_inverse_softabs_hessian
#             current_inverse_softabs_hessian .= proposal_inverse_softabs_hessian
#             current_softabs_hessian .= proposal_softabs_hessian
#             accepted_moves += 1
#         end
#
#         if iteration_index%thinning_rate == 0
#             mcmc_samples[Int64(iteration_index/thinning_rate),:] = current_position
#         end
#     end #main for loop
#
#     println(string("Acceptance ratio: ",accepted_moves/number_of_iterations))
#     return mcmc_samples
# end
