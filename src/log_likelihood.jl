"""
Calculates the negative log of the likelihood of our data given the paramters, using the Kalman filter. It uses the
predicted_observation_distributions from the kalman_filter function. The entries of this array in the second and
third columns represent the probability of the future observation of mRNA and Protein respectively, given our current knowledge.

Parameters
----------

protein_at_observations : numpy array.
    Observed protein. The dimension is m x n x 2, where m is the number of data sets, n is the
    number of observation time points. For each data set, the first column is the time,
    and the second column is the observed protein copy number at that time.

model_parameters : numpy array.
    An array containing the model parameters in the following order:
    repression_threshold, hill_coefficient, mRNA_degradation_rate,
    protein_degradation_rate, basal_transcription_rate, translation_rate,
    transcription_delay.

measurement_variance : float.
    The variance in our measurement. This is given by Sigma_e in Calderazzo et. al. (2018).

Returns
-------

log_likelihood : float.
    The log of the likelihood of the data.
"""
function calculate_log_likelihood_at_parameter_point(protein_at_observations,model_parameters,measurement_variance = 10)
    if any(model_parameters .< 0)
        return -Inf
    end
    _, _, predicted_observation_distributions = kalman_filter(protein_at_observations,
                                                              model_parameters,
                                                              measurement_variance)
    observations = protein_at_observations[:,2]
    mean = predicted_observation_distributions[:,2]
    sd = sqrt.(predicted_observation_distributions[:,3])

    log_likelihood = sum([logpdf(Normal(mean[i],sd[i]),observations[i]) for i in 1:length(observations)])
    # logpdf(Normal(mean[1],sd[1]),observations[1])
    # for observation_index in 2:length(observations)
    #     log_likelihood += logpdf(Normal(mean[observation_index],sd[observation_index]),observations[observation_index])
    # end #for
    # return log_likelihood
end # function


"""
Calculates the log of the likelihood, and the derivative of the negative log likelihood wrt each parameter, of our data given the
paramters, using the Kalman filter. It uses the predicted_observation_distributions, predicted_observation_mean_derivatives, and
predicted_observation_variance_derivatives from the kalman_filter function. It returns the log likelihood as in the
calculate_log_likelihood_at_parameter_point function, and also returns an array of the derivative wrt each parameter.

Parameters
----------

protein_at_observations : numpy array.
    Observed protein. The dimension is n x 2, where n is the number of observation time points.
    The first column is the time, and the second column is the observed protein copy number at
    that time.

model_parameters : numpy array.
    An array containing the moderowl parameters in the following order:
    repression_threshold, hill_coefficient, mRNA_degradation_rate,
    protein_degradation_rate, basal_transcription_rate, translation_rate,
    transcription_delay.

measurement_variance : float.
    The variance in our measurement. This is given by Sigma_e in Calderazzo et. al. (2018).

Returns
-------

log_likelihood : float.
    The log of the likelihood of the data.

log_likelihood_derivative : numpy array.
    The derivative of the log likelihood of the data, wrt each model parameter
"""
function calculate_log_likelihood_derivative_at_parameter_point(protein_at_observations,model_parameters,measurement_variance)
    f_test = x -> calculate_log_likelihood_at_parameter_point(protein_at_observations,x,measurement_variance) # create single argument function
    g_test = x -> ForwardDiff.gradient(f_test, x)
    g_test(model_parameters)
end #function
