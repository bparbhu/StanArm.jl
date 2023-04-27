using Stan
using Pkg.Artifacts

function precompile_model(model_name::String, model_path::String)
    # Read the Stan model code
    model_code = read(model_path, String)

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
    ("stan_glm", "src/stan_files/models/stan_glm.stan")
]

artifacts = Dict{String, String}()

for (model_name, model_path) in models
    artifacts[model_name] = precompile_model(model_name, model_path)
end

# Save the artifact mapping
Pkg.Artifacts.toml_save("Artifacts.toml", artifacts)
