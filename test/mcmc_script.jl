using DelayedKalmanFilter
using Turing
using DelimitedFiles

loading_path = string(pwd(),"/data/");
protein_observations = readdlm(string(loading_path,"kalman_filter_test_trace_observations.csv"),',');
model_parameters = [10000.0,5.0,log(2)/30,log(2)/90,1.0,1.0,29.0];
measurement_variance = 10000;

f_test = x -> calculate_log_likelihood_at_parameter_point(protein_observations,x,measurement_variance)

@model function full_kalman_model(protein_observations,::Type{T} = Float64) where {T}
          model_parameters = Vector{T}(undef,7)
          model_parameters[1] ~ Uniform(500,2*mean(protein_observations[:,2])) #repression_threshold
          model_parameters[2] ~ Uniform(2,6) #hill_coefficient
          model_parameters[3] ~ Uniform(log(2)/200,log(2)/10) #mRNA_degradation_rate
          model_parameters[4] ~ Uniform(log(2)/200,log(2)/10) #protein_degradation_rate
          model_parameters[5] ~ Uniform(0.01,120) #basal_transcription_rate
          model_parameters[6] ~ Uniform(0.01,40) #translation_rate
          model_parameters[7] ~ Uniform(1,40) #transcriptional_delay
          Turing.@addlogprob! f_test(model_parameters)
      end
      
      
c1 = sample(full_kalman_model(protein_observations),NUTS(0.65),MCMCThreads(),100,4)


# ignore time
@model function full_kalman_model(protein_observations,::Type{T} = Float64) where {T}
          model_parameters = Vector{T}(undef,7)
          model_parameters[7] = 29.0 #transcriptional_delay
          model_parameters[3] = log(2)/30 #mRNA_degradation_rate
          model_parameters[4] = log(2)/90 #protein_degradation_rate
          
          model_parameters[1] ~ Uniform(1500,2*mean(protein_observations[:,2])) #repression_threshold
          model_parameters[2] ~ Uniform(2,6) #hill_coefficient
          model_parameters[5] ~ Uniform(log(0.01),log(120)) #basal_transcription_rate
          model_parameters[6] ~ Uniform(log(0.01),log(40)) #translation_rate
          model_parameters[[5,6]] .= exp.(model_parameters[[5,6]])
          Turing.@addlogprob! f_test(model_parameters)
      end
