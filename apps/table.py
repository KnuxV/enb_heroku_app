from dash import html, dcc
from dash.dependencies import Input, Output
import pandas as pd
import pathlib
from app import app
from prep_functions import upperpart_layout, \
    generate_table, filter_db, update_markdown

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()
df_ini = pd.read_pickle(DATA_PATH.joinpath("df_all_message_venturini_categorical.pkl"))
# df_ini = pd.read_pickle(DATA_PATH.joinpath("df_full_enb_corpus_categorical.pkl"))

layout = html.Div(children=[
    html.Div([
        html.Button("Download CSV", id="btn_csv"),
        dcc.Download(id="download-dataframe-csv"),
        html.H1('Summary', style={"textAlign": "center"}),
    ]),
    html.Div(id='upper', children=[
        upperpart_layout(df_ini, page='summary-markdown-table'),
    ]),
    html.H5(id="table-with-slider", children='ENB Preview')
])


@app.callback(
    [Output('table-with-slider', 'children'),
     Output('summary-markdown-table', 'children')],
    [Input('range-slider', 'value'),
     Input('country_input', 'value'),
     Input('search_input', 'value'),
     Input('keyword_input', 'value')],
)
def update_table(selected_range, countries, search, keywords):
    filtered_df = filter_db(df_ini, selected_range=selected_range, countries=countries, search=search,
                            keywords=keywords)

    return generate_table(filtered_df, max_rows=len(filtered_df)), update_markdown(filtered_df)


@app.callback(
    Output("download-dataframe-csv", "data"),
    [Input("btn_csv", "n_clicks"),
     Input('range-slider', 'value'),
     Input('country_input', 'value'),
     Input('search_input', 'value'),
     Input('keyword_input', 'value')],

    prevent_initial_call=True,
)
def func(n_clicks, selected_range, countries, search, keywords):
    if n_clicks:
        filtered_df = filter_db(df_ini, selected_range=selected_range, countries=countries, search=search,
                                keywords=keywords)
        return dcc.send_data_frame(filtered_df.to_csv, "mydf.csv")
