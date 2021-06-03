using DelayedKalmanFilter
using DelimitedFiles
using Test

@testset "MCMC tests" begin
    loading_path = string(pwd(),"/data/");
    protein_at_observations = readdlm(string(loading_path,"kalman_filter_test_trace_observations.csv"),',');
    model_parameters = [10000.0,5.0,log(2)/30,log(2)/90,1.0,1.0,29.0];
    measurement_variance = 10000;
    number_of_samples = 10;

    output = kalman_filter_mh(protein_at_observations,measurement_variance,number_of_samples);
    @test size(output.value.data[:,:,1]) == (number_of_samples,length(model_parameters)+1-2) # includes log prob

    output = kalman_filter_mh(protein_at_observations,measurement_variance,number_of_samples; full_model=true);
    @test size(output.value.data[:,:,1]) == (number_of_samples,length(model_parameters)+1) # includes log prob

    output = kalman_filter_NUTS(protein_at_observations,measurement_variance,number_of_samples);
    @test size(output.value.data[:,:,1]) == (number_of_samples,length(model_parameters)+12-2) # includes log prob and extra things (tree depth etc)

    output = kalman_filter_NUTS(protein_at_observations,measurement_variance,number_of_samples; full_model=true);
    @test size(output.value.data[:,:,1]) == (number_of_samples,length(model_parameters)+12) # includes log prob and extra things (tree depth etc)
end
