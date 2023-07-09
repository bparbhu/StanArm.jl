using Stan
using Pkg.Artifacts

function stan_models()
    # Assume that models_home variable is the path to the directory where the Stan files are located
    models_home = "src"
    stan_files = filter(s -> endswith(s, ".stan"), readdir(models_home))

    stanmodels = Dict()

    for stan_file in stan_files
        model_cppname = splitext(stan_file)[1]
        stanmodels[model_cppname] = nothing
    end

    # Now let's compile and sample from each model.
    for (model_name, _) in stanmodels
        stan_code = read(joinpath(models_home, model_name * ".stan"), String)
        stanmodel = Stanmodel(name=model_name, model=stan_code)
        # Replace the `data` argument below with the actual data for the model
        _, chn = stan(stanmodel, data)
        stanmodels[model_name] = chn
    end


function precompile_model(model_name::String, model_path::String)
    # Read the Stan model code
    model_code = join(readlines(model_path), "\n")

    # Create a Stan model and compile it
    model = Stanmodel(name=model_name, model=model_code, output_format=:mcmcchains)
    build(model)

    # Create an artifact for the compiled model
    model_dir = abspath(dirname(model_path))
    artifact_dir = create_artifact() do artifact_path
        cp(joinpath(model_dir, "$(model_name).$(Libdl.dlext)"), artifact_path)
    end

    return artifact_dir
end

models = [
    ("bernoulli", "src/stan_files/bernoulli.stan")
]

artifacts = Dict{String, String}()

for (model_name, model_path) in models
    artifacts[model_name] = precompile_model(model_name, model_path)
end

# Save the artifact mapping
Pkg.Artifacts.toml_save("Artifacts.toml", artifacts)
