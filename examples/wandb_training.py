import d3rlpy

if __name__ == "__main__":
    dataset, env = d3rlpy.datasets.get_cartpole()

    dqn = d3rlpy.algos.DQNConfig().create()

    team="rl-il"
    project ="bench-rlil"
    dqn.fit(
        dataset=dataset,
        n_steps=100000,
        # set FileAdapterFactory to save metrics as CSV files
        logger_adapter=d3rlpy.logging.WanDBAdapterFactory(project=project),
    )

