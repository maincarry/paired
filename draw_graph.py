import seaborn as sns
import pandas as pd

exp_name = "ued-Gfootball-test-DR-pass_n_shoot_v2"
metric = "mean_agent_return"

path_to_csv = f"~/logs/paired/{exp_name}/logs.csv"
out_name = path_to_csv.split("/")[-2] + ".png"

data = pd.read_csv(path_to_csv)
# print(data.columns)
sns_plot = sns.relplot(x="steps", y=metric, data=data)
# sns_plot = sns.relplot(x="steps", y="agent_pg_loss", data=data)
# sns_plot = sns.relplot(x="steps", y="agent_value_loss", data=data)
sns_plot.fig.suptitle(f"{exp_name}:{metric}")
sns_plot.savefig(out_name)



