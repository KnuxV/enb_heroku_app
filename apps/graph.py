from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
import pathlib
from app import app
from prep_functions import country_repartition, keyword_repartition, time_evolution, \
    upperpart_layout, network_graph, filter_db, update_markdown, type_predicate_repartition

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()

df_ini = pd.read_pickle(DATA_PATH.joinpath("df_all_message_venturini_categorical.pkl"))

layout = html.Div(children=[
    html.Div(id='upper', children=[
        upperpart_layout(df_ini, page='summary-markdown-graph'),
    ]),
    html.Div(id='threegraphs', className='row', children=[
        html.Div(id='worldgrah', children=[
            # dcc.Graph(id='graph', figure=network_graph(df_ini, 20))
        ]),
        html.Div(id='predgraph', className='row', children=[
            html.Div(className='six columns', children=[
                dcc.RadioItems(
                    id='radio-nb-country',
                    options=[
                        {'label': '5', 'value': 5},
                        {'label': '10', 'value': 10},
                        {'label': '20', 'value': 20}
                    ],
                    value=10,
                    labelStyle={'display': 'inline-block'}
                ),
                html.Div(id='country_repartition', children=[
                    dcc.Markdown("country_repartition")
                ])

            ]),
            html.Div(className='six columns', children=[
                html.Div(id='predicate_repartition', children=[
                    dcc.Markdown("predicate_repartition")
                ])
            ])
        ]),
        html.Div(id='othergraph', className='row', children=[

            html.Div(className='six columns', children=[

                html.Div(id='keyword_repartition', children=[

                ]),

            ]),
            html.Div(className='six columns', children=[
                dcc.RadioItems(
                    id='time-evolution',
                    options=[
                        {'label': 'country', 'value': 'country'},
                        {'label': 'cluster', 'value': 'cluster'}
                    ],
                    value='country',
                    labelStyle={'display': 'inline-block'}
                ),

                html.Div(id='year-evolution', children=[

                ]),
            ])
        ])
    ])
])


@app.callback(
    [Output('country_repartition', 'children'),
     Output('keyword_repartition', 'children'),
     Output('year-evolution', 'children'),
     Output('worldgrah', 'children'),
     Output('predicate_repartition', 'children'),
     Output('summary-markdown-graph', 'children')
     ],
    [Input('range-slider', 'value'),
     Input('country_input', 'value'),
     Input('search_input', 'value'),
     Input('keyword_input', 'value'),
     Input('radio-nb-country', 'value'),
     Input('time-evolution', 'value'),
     Input('predicate_input', 'value')
     ],

)
def update_page(selected_range, countries, search, keywords, nb_country, typ, predicate):
    filtered_df = filter_db(df_ini, selected_range=selected_range, countries=countries, search=search,
                            keywords=keywords, predicate=predicate)

    return_country_repartition = dcc.Graph(figure=country_repartition(filtered_df, nb_country=nb_country))
    return_keyword_repartition = dcc.Graph(id="test", figure=keyword_repartition(filtered_df))
    return_year_evolution = dcc.Graph(figure=time_evolution(filtered_df, typ=typ))
    return_network_graph = dcc.Graph(id='graph', figure=network_graph(filtered_df, 20))
    return_predicate_repartition = dcc.Graph(id='pred_graph', figure=type_predicate_repartition(filtered_df))

    return return_country_repartition, return_keyword_repartition, return_year_evolution, return_network_graph, \
           return_predicate_repartition,update_markdown(filtered_df)
