import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import dash_auth

VALID_USERNAME_PASSWORD_PAIRS = {
    "ozaj": "Jeet1992"
}
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)
def z_score(isin, curr_value, col):
    df = pd.read_csv(r"C:\Jeet\Data\Database\%s.csv" % isin, index_col='Date')
    vals = df[col]
    return (curr_value - vals.mean()) / vals.std()


def hist_mean(isin, col):
    df = pd.read_csv(r"C:\Jeet\Data\Database\%s.csv" % isin, index_col='Date')
    vals = df[col]
    return vals.mean()

def hist_std(isin, col):
    df = pd.read_csv(r"C:\Jeet\Data\Database\%s.csv" % isin, index_col='Date')
    vals = df[col]
    return vals.std()

def hist_max(isin, col):
    df = pd.read_csv(r"C:\Jeet\Data\Database\%s.csv" % isin, index_col='Date')
    vals = df[col]
    return vals.max()

def hist_min(isin, col):
    df = pd.read_csv(r"C:\Jeet\Data\Database\%s.csv" % isin, index_col='Date')
    vals = df[col]
    return vals.min()

df = pd.read_excel(r"C:\Jeet\Data\securities\Ukraine curves all bonds.xlsm", sheet_name='UKRAINE MONITOR', skiprows=8)
for col in ["LAST", "ASW FX", "ASW$"]:
    df["%s.Z_score"%col] = df.apply(lambda x:z_score(x.ISIN, x[col],col), axis = 1)
    df["%s.Mean"%col] = df.apply(lambda x: hist_mean(x.ISIN, col), axis=1)
    df["%s.Std" % col] = df.apply(lambda x: hist_std(x.ISIN, col), axis=1)
df = df.drop(df.columns[0], axis=1).round(3)

ISINs = df.ISIN
cols = ["LAST", "YTM", "ASW FX", "ASW$", "DUR", "IA", "DIRTY", "FX"]

hist_layout = html.Div([
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='ISIN-Val',
                options=[{'label': i, 'value': i} for i in ISINs],
                value=ISINs[0]
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                id='Indicator-Val',
                options=[{'label': i, 'value': i} for i in cols],
                value=cols[0]
            )
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ]),

    dcc.Graph(id='Historic-Plot'),

    dash_table.DataTable(
        id='table',
        merge_duplicate_headers=True,
        columns=[{"name": i.split("."), "id": i,"hideable": True} for i in df.columns],
        data=df.to_dict("rows"),
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ],
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        }
    )
])

comp_layout = html.Div([
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='ISIN-Val1',
                options=[{'label': i, 'value': i} for i in ISINs],
                value=ISINs[0]
            ),
            dcc.Dropdown(
                id='ISIN-Val2',
                options=[{'label': i, 'value': i} for i in ISINs],
                value=ISINs[1]
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                id='Indicator-Val1',
                options=[{'label': i, 'value': i} for i in cols],
                value=cols[0]
            )
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ]),

    html.Div([
        html.Div([
            dcc.Graph(id='Comp-Plot')
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(id='Diff-Plot')
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ]),

    html.Div([
        html.Div([
            dash_table.DataTable(id='Comp-table', merge_duplicate_headers=True,
                 filter_action="native",
                 sort_action="native",
                 sort_mode="multi",
                 style_data_conditional=[
                     {
                         'if': {'row_index': 'odd'},
                         'backgroundColor': 'rgb(248, 248, 248)'
                     }
                 ],
                 style_header={
                     'backgroundColor': 'rgb(230, 230, 230)',
                     'fontWeight': 'bold'
                 }
            )
        ], style={'width': '58%', 'display': 'inline-block'}),
        html.Div([
            dash_table.DataTable(id='Stat-table', merge_duplicate_headers=True,
                 filter_action="native",
                 sort_action="native",
                 sort_mode="multi",
                 style_data_conditional=[
                     {
                         'if': {'row_index': 'odd'},
                         'backgroundColor': 'rgb(248, 248, 248)'
                     }
                 ],
                 style_header={
                     'backgroundColor': 'rgb(230, 230, 230)',
                     'fontWeight': 'bold'
                 }
             )
        ], style={'width': '38%', 'float': 'right', 'display': 'inline-block'})
    ])

])

app.layout = html.Div([
    dcc.Tabs(id="tabs", value='Monitor1', children=[
        dcc.Tab(label='Monitor', value='Monitor1'),
        dcc.Tab(label='Comparison', value='Comparison1'),
    ]),
    html.Div(id='tabs-content')
])


@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'Monitor1':
        return hist_layout
    elif tab == 'Comparison1':
        return comp_layout


@app.callback(
    Output('Historic-Plot', 'figure'),
    [Input('ISIN-Val', 'value'),
     Input('Indicator-Val', 'value')])
def update_graph(ISIN, Col):
    df = pd.read_csv(r"C:\Jeet\Data\Database\%s.csv" % ISIN)
    df.Date = pd.to_datetime(df.Date)
    return {
        'data': [go.Scatter(
            x=df['Date'],
            y=df[Col],
            mode='lines+markers'
        )],
        'layout': go.Layout(
            xaxis={
                'title': "Date",
                'type': 'date'
            },
            yaxis={
                'title': Col,
                'type': 'linear'
            },
            margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
            hovermode='closest'
        )
    }


@app.callback([
    Output('Comp-table', 'data'),
    Output('Comp-table', 'columns')],
    [Input('ISIN-Val1', 'value'),
     Input('ISIN-Val2', 'value'),
     Input('Indicator-Val1', 'value')])
def update_graph(ISIN1, ISIN2, Col):
    df1 = pd.read_csv(r"C:\Jeet\Data\Database\%s.csv" % ISIN1)
    df2 = pd.read_csv(r"C:\Jeet\Data\Database\%s.csv" % ISIN2)
    dfC = pd.concat([df1, df2])
    dfP = dfC.pivot(index="Date", columns="ISIN", values=["LAST", "ASW FX", "ASW$"])
    dfP = dfP.sort_index(ascending=False)
    dfP.columns = ["%s.%s" % (x[0], x[1]) for x in dfP.columns]
    dfP = dfP.reset_index().round(3)
    return dfP.to_dict("rows"), [{"name": i.split("."), "id": i, "hideable": True} for i in dfP.columns]


@app.callback([
    Output('Stat-table', 'data'),
    Output('Stat-table', 'columns')],
    [Input('ISIN-Val1', 'value'),
     Input('ISIN-Val2', 'value')])
def update_graph(ISIN1, ISIN2):
    df = pd.read_excel(r"C:\Jeet\Data\securities\Ukraine curves all bonds.xlsm", sheet_name='UKRAINE MONITOR',
                       skiprows=8)
    for col in ["LAST", "ASW FX", "ASW$"]:
        df["%s.Z_score" % col] = df.apply(lambda x: z_score(x.ISIN, x[col], col), axis=1)
        df["%s.Mean" % col] = df.apply(lambda x: hist_mean(x.ISIN, col), axis=1)
        df["%s.Std" % col] = df.apply(lambda x: hist_std(x.ISIN, col), axis=1)
        df["%s.Max" % col] = df.apply(lambda x: hist_max(x.ISIN, col), axis=1)
        df["%s.Min" % col] = df.apply(lambda x: hist_min(x.ISIN, col), axis=1)
    df = df.drop(df.columns[0], axis=1).round(3)
    df1 = pd.DataFrame()
    for col in ["LAST", "ASW FX", "ASW$"]:
        for isin in [ISIN1, ISIN2]:
            df1 = df1.append({"ISIN": isin, "Col": col, "Method": "Current",
                              "Value": df[df.ISIN == isin][col].values[0]}, ignore_index=True)
            for method in ["Z_score", "Mean", "Std", "Max", "Min"]:
                df1 = df1.append({"ISIN" : isin, "Col" : col, "Method" : method,
                                  "Value" : df[df.ISIN==isin]["%s.%s"%(col,method)].values[0]}, ignore_index = True)
    df2 = df1.pivot_table(values="Value", index="Method", columns=["ISIN", "Col"])
    df2.columns = ["%s.%s" % (x[0], x[1]) for x in df2.columns]
    df2 = df2.reset_index().round(3)
    return df2.to_dict("rows"), [{"name": i.split("."), "id": i, "hideable": True} for i in df2.columns]

@app.callback([
    Output('table', 'data'),
    Output('table', 'columns')],
    [])
def update_graph():
    df = pd.read_excel(r"C:\Jeet\Data\securities\Ukraine curves all bonds.xlsm", sheet_name='UKRAINE MONITOR',
                       skiprows=8)
    for col in ["LAST", "ASW FX", "ASW$"]:
        df["%s.Z_score" % col] = df.apply(lambda x: z_score(x.ISIN, x[col], col), axis=1)
        df["%s.Mean" % col] = df.apply(lambda x: hist_mean(x.ISIN, col), axis=1)
        df["%s.Std" % col] = df.apply(lambda x: hist_std(x.ISIN, col), axis=1)
    df = df.drop(df.columns[0], axis=1).round(3)
    return df.to_dict("rows"), [{"name": i.split("."), "id": i, "hideable": True} for i in df.columns]


@app.callback(Output('Comp-Plot', 'figure'),
              [Input('ISIN-Val1', 'value'),
               Input('ISIN-Val2', 'value'),
               Input('Indicator-Val1', 'value')])
def update_graph(ISIN1, ISIN2, Col):
    df1 = pd.read_csv(r"C:\Jeet\Data\Database\%s.csv" % ISIN1)
    df2 = pd.read_csv(r"C:\Jeet\Data\Database\%s.csv" % ISIN2)
    dfC = pd.concat([df1, df2])
    dfP = dfC.pivot(index="Date", columns="ISIN", values=Col)
    dfP = dfP.reset_index()
    dfP.Date = pd.to_datetime(dfP.Date)
    dfP = dfP.dropna()
    return {
        'data': [
            go.Scatter(
                x=dfP['Date'],
                y=dfP[ISIN1],
                name=ISIN1,
                mode='lines+markers'
            ),
            go.Scatter(
                x=dfP['Date'],
                y=dfP[ISIN2],
                name=ISIN2,
                mode='lines+markers'
            )
        ],
        'layout': go.Layout(
            xaxis={
                'title': "Date",
                'type': 'date'
            },
            yaxis={
                'title': Col,
                'type': 'linear'
            },
            margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
            hovermode='closest'
        )
    }


@app.callback(Output('Diff-Plot', 'figure'),
              [Input('ISIN-Val1', 'value'),
               Input('ISIN-Val2', 'value'),
               Input('Indicator-Val1', 'value')])
def update_graph(ISIN1, ISIN2, Col):
    df1 = pd.read_csv(r"C:\Jeet\Data\Database\%s.csv" % ISIN1)
    df2 = pd.read_csv(r"C:\Jeet\Data\Database\%s.csv" % ISIN2)
    dfC = pd.concat([df1, df2])
    dfP = dfC.pivot(index="Date", columns="ISIN", values=Col)
    dfP['Diff'] = dfP[ISIN1] - dfP[ISIN2]
    dfP = dfP.reset_index()
    dfP.Date = pd.to_datetime(dfP.Date)
    dfP = dfP.dropna()
    return {
        'data': [
            go.Scatter(
                x=dfP['Date'],
                y=dfP['Diff'],
                name="Diff %s" % Col,
                mode='lines+markers'
            )
        ],
        'layout': go.Layout(
            xaxis={
                'title': "Date",
                'type': 'date'
            },
            yaxis={
                'title': "Diff %s" % Col,
                'type': 'linear'
            },
            margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
            hovermode='closest'
        )
    }


if __name__ == '__main__':
    app.run_server(debug=True)
