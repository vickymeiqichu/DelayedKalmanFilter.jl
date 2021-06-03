using DelayedKalmanFilter
using DelimitedFiles
using Test

@testset "Likelihood tests" begin
    loading_path = string(pwd(),"/data/");
    protein_at_observations = readdlm(string(loading_path,"kalman_filter_test_trace_observations.csv"),',');
    model_parameters = [10000.0,5.0,log(2)/30,log(2)/90,1.0,1.0,29.0];
    measurement_variance = 10000;

    # test negative parameter check works
    negative_parameter_1 = [10000.0,5.0,log(2)/30,log(2)/90,-1.0,1.0,29.0];
    negative_parameter_2 = [-10000.0,5.0,log(2)/30,log(2)/90,1.0,1.0,29.0];

    ll = calculate_log_likelihood_at_parameter_point(protein_at_observations,negative_parameter_1,measurement_variance);
    @test ll == -Inf
    ll = calculate_log_likelihood_at_parameter_point(protein_at_observations,negative_parameter_2,measurement_variance);
    @test ll == -Inf
    # test log likelihood and derivative are correct
    ll = calculate_log_likelihood_at_parameter_point(protein_at_observations,model_parameters,measurement_variance);
    @test ll ≈ readdlm(string(loading_path,"log_likelihood_value.csv"),',')[1]

    lld = calculate_log_likelihood_derivative_at_parameter_point(protein_at_observations,model_parameters,measurement_variance);
    @test all(lld .== readdlm(string(loading_path,"log_likelihood_auto_derivative_value.csv"),','))

    hessian = calculate_log_likelihood_hessian_at_parameter_point(protein_at_observations,model_parameters,measurement_variance);
    @test size(hessian) == (length(model_parameters),length(model_parameters))
end
