import seaborn as sns
import pandas as pd

exp_name = "ued-Gfootball-test-DR-avoid_pass_shoot_v2_1"
metric = "mean_agent_return" # "agent_pg_loss" "agent_value_loss"

path_to_csv = f"~/logs/paired/{exp_name}/logs.csv"
out_name = path_to_csv.split("/")[-2] + "-" + metric + ".png"

data = pd.read_csv(path_to_csv)
# print(data.columns)
sns_plot = sns.relplot(x="steps", y=metric, data=data)
# sns_plot = sns.relplot(x="steps", y="agent_pg_loss", data=data)
# sns_plot = sns.relplot(x="steps", y="agent_value_loss", data=data)
sns_plot.fig.suptitle(f"{exp_name}:{metric}")
sns_plot.savefig(out_name)



