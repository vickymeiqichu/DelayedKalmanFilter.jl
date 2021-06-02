"""
todo
"""
function kalman_filter_mh(protein_observations,model_parameters,measurement_variance,number_of_samples; full_model=false)

   if full_model
      @model full_kalman_model(protein_observations) = begin
             model_parameters[1] ~ Uniform(5000,15000) #repression_threshold
             model_parameters[2] ~ Uniform(2,6) #hill_coefficient
             model_parameters[3] ~ Uniform(log(2)/200,log(2)/10) #mRNA_degradation_rate
             model_parameters[4] ~ Uniform(log(2)/200,log(2)/10) #protein_degradation_rate
             model_parameters[5] ~ Uniform(0.01,120) #basal_transcription_rate
             model_parameters[6] ~ Uniform(0.01,40) #translation_rate
             model_parameters[7] ~ Uniform(1,40) #transcriptional_delay
             Turing.@addlogprob! calculate_log_likelihood_at_parameter_point(protein_observations,
                                                                             model_parameters,
                                                                             measurement_variance)
         end

      c1 = sample(full_kalman_model(protein_observations),MH(diagm(fill(1,7))),number_of_samples);
      return c1
   end #if

    @model kalman_model(protein_observations) = begin
           model_parameters[1] ~ Uniform(5000,15000) #repression_threshold
           model_parameters[2] ~ Uniform(2,6) #hill_coefficient
           model_parameters[5] ~ Uniform(0.01,120) #basal_transcription_rate
           model_parameters[6] ~ Uniform(0.01,40) #translation_rate
           model_parameters[7] ~ Uniform(1,40) #transcriptional_delay
           Turing.@addlogprob! calculate_log_likelihood_at_parameter_point(protein_observations,
                                                                           model_parameters,
                                                                           measurement_variance)
       end

    c1 = sample(kalman_model(protein_observations),MH(diagm(fill(1,5))),number_of_samples);
    return c1
end #function

"""
todo
"""
function kalman_filter_NUTS(protein_observations,model_parameters,measurement_variance,number_of_samples; full_model=false)
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

      c1 = sample(full_kalman_model(protein_observations),NUTS(0.65),number_of_samples);
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

   c1 = sample(kalman_model(protein_observations),NUTS(0.65),number_of_samples);
   return c1
end #function