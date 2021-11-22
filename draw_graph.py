import seaborn as sns
import pandas as pd

path_to_csv = "~/logs/paired/ued-Gfootball-test-DR-offense-v2/logs.csv"
out_name = path_to_csv.split("/")[-2] + ".png"

data = pd.read_csv(path_to_csv)
# print(data.columns)
sns_plot = sns.relplot(x="steps", y="mean_agent_return", data=data)
# sns_plot = sns.relplot(x="steps", y="agent_pg_loss", data=data)
# sns_plot = sns.relplot(x="steps", y="agent_value_loss", data=data)
sns_plot.savefig(out_name)



