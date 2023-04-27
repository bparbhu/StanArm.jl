module StanArm

using Stan
using DataFrames
using Artifacts


#Data
include("stan_files/data/data_assoc.stan")
include("stan_files/data/data_betareg.stan")
include("stan_files/data/data_event.stan")
include("stan_files/data/data_glm.stan")
include("stan_files/data/data_mvmer.stan")
include("stan_files/data/dimensions_mvmer.stan")
include("stan_files/data/glmer_stuff.stan")
include("stan_files/data/glmer_stuff2.stan")
include("stan_files/data/hyperparameters_assoc.stan")
include("stan_files/data/hyperparameters_event.stan")
include("stan_files/data/hyperparameters_mvmer.stan")


# Load the artifact mapping
artifact_dir = abspath(joinpath(@__DIR__, "..", "Artifacts.toml"))
artifacts = Pkg.Artifacts.artifact_paths(artifact_dir)

# Function to get the precompiled model path
function get_precompiled_model_path(model_name::String)
    return artifacts[model_name]
end


function get_stan_file_content(file_name::String)
    stan_file_path = joinpath(@__DIR__, "stan_files", file_name)
    return read(stan_file_path, String)
end


function stan_glm(formula::FormulaTerm, data::DataFrame; kwargs...)
    # Get the Stan model code
    stan_code = get_stan_file_content("stan_glm.stan")

    # Replace the @include statements with the actual content of the included files
    stan_code = replace(stan_code, r"@include (\w+\.stan)" => (m) -> get_stan_file_content(m[1]))

    # Get the precompiled model path
    precompiled_model_path = get_precompiled_model_path("stan_glm")

    # Prepare data for Stan
    stan_data = Dict{String, Any}(
        # Map your data from DataFrame to the Stan model's data block
    )

    # Call the Stan function using the precompiled model
    model = Stanmodel(name="stan_glm", model_path=precompiled_model_path, model_code=stan_code, output_format=:mcmcchains)
    fit = stan(model, stan_data; kwargs...)

    return fit
end
