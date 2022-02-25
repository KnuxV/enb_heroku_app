import pathlib
import typing
from collections import Counter
from itertools import combinations, product

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import html, dcc

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("datasets").resolve()

df_ini = pd.read_pickle(DATA_PATH.joinpath("df_all_message_venturini.pkl"))


def filter_db(df, selected_range, countries, search, keywords):
    cond_on_year = df['Year'].between(selected_range[0], selected_range[1], inclusive='both')
    filtered_df = df[cond_on_year]
    if countries:
        for country in countries:
            if filtered_df['Actor_Support'].str.contains(country).any():
                cond_on_country = filtered_df['Actor_Support'].str.contains(country)
                filtered_df = filtered_df[cond_on_country]
    # SEARCH LOGIC
    if search != "":
        cond_search = filtered_df["Message"].str.contains(search)
        filtered_df = filtered_df[cond_search]
    for keyword in keywords:
        if filtered_df['Venturini_keywords'].str.contains(keyword).any():
            cond_on_keyword = filtered_df['Venturini_keywords'].str.contains(keyword)
            filtered_df = filtered_df[cond_on_keyword]
    return filtered_df


def main_countries(df, nb_country=10):
    c_sup = Counter()
    for ind, row in df.iterrows():
        lst_actor_sup = row.Actor_Support.split(', ')
        for actor in lst_actor_sup:
            c_sup[actor] += 1
    del c_sup[""]
    return [tup[0] for tup in c_sup.most_common(nb_country)]


def country_repartition(df, nb_country=10):
    c_sup = Counter()
    c_opp = Counter()
    for ind, row in df.iterrows():
        lst_actor_sup = row.Actor_Support.split(', ')
        lst_actor_opp = row.Actor_Opposition.split(', ')
        for actor in lst_actor_sup:
            c_sup[actor] += 1
        for actor in lst_actor_opp:
            c_opp[actor] += 1
    del c_sup[""]
    del c_opp[""]
    m_c = [tup[0] for tup in (c_sup + c_opp).most_common(nb_country)]
    res = []
    for country in m_c:
        tup1 = (country, 'opposition', c_opp[country])
        tup2 = (country, 'support', c_sup[country])
        res.append(tup1)
        res.append(tup2)

    df_res = pd.DataFrame(res, columns=['Country', 'Support', 'Mention'])
    color_discrete_map = {'opposition': 'rgb(255,0,0)', 'support': 'rgb(0,0,255)'}
    fig = px.bar(df_res, x="Country", y="Mention", color="Support", text_auto=True,
                 color_discrete_map=color_discrete_map, title="Country participation in mentions")
    return fig


def main_keyword(df):
    lst_keywords = unique_keywords(df)
    ck = Counter()
    for keyword_in_lst in lst_keywords:
        ck[keyword_in_lst] = 0
    for ind, row in df.iterrows():
        for keyword_in_col in row.Venturini_keywords.split(', '):
            if keyword_in_col in lst_keywords:
                ck[keyword_in_col] += 1
    return ck.most_common()


def keyword_repartition(df):
    lst_main_keywords = main_keyword(df)
    df_keywords_counter = pd.DataFrame(lst_main_keywords, columns=['keyword', 'count'])
    # fig = px.bar(df_keywords_counter, x="keyword", y="count", text_auto=True)
    fig = px.pie(df_keywords_counter, names="keyword", values="count",
                 title="Frequency of occurrence of Venturini clusters")
    fig.update_traces(textposition='inside', textinfo='value+percent')
    fig.update_layout(title_font_size=12,
                      legend=dict(orientation="h", font=dict(family="Courier", size=10, color="black")),
                      legend_title=dict(font=dict(family="Courier", size=10, color="blue")))

    return fig


def time_evolution(df, typ="country", nb_item=5):
    if typ == "country":
        m_c = main_countries(df, nb_item)
        dic_year = {y: Counter() for y in range(1995, 2020)}
        for ind, row in df.iterrows():
            year = int(row.Year)
            for country in row.Actor_Support.split(', ') + row.Actor_Opposition.split(', '):
                if country in m_c:
                    dic_year[year][country] += 1
        lst_lst = []
        for year in dic_year:
            for country, count in dic_year[year].items():
                tup = (year, country, count)
                lst_lst.append(tup)

        df_country_year = pd.DataFrame(lst_lst, columns=['year', 'country', 'count'])
        fig = px.line(df_country_year, x="year", y="count", color='country')
        return fig
    if typ == "keyword":
        m_k = [tup[0] for tup in main_keyword(df)]
        dic_year = {y: Counter() for y in range(1995, 2020)}
        for ind, row in df.iterrows():
            year = int(row.Year)
            for keyword in row.Venturini_keywords.split(', '):
                if keyword in m_k:
                    dic_year[year][keyword] += 1
        lst_lst = []
        for year in dic_year:
            for keyword, count in dic_year[year].items():
                tup = (year, keyword, count)
                lst_lst.append(tup)
        df_keyword_year = pd.DataFrame(lst_lst, columns=['year', 'keyword', 'count'])
        fig = px.line(df_keyword_year, x="year", y="count", color='keyword')
        return fig


def unique_actors(df, support="all") -> typing.List:
    """
    return a list with all unique actors in the df
    :param support:
        Either "all", "sup", or "opp"
    :param df:
    :return:
    """
    typ = "triple" if "Actor" in df.columns else "message"
    return_list = []
    if typ == "triple":
        for ind, row in df.iterrows():
            return_list += row.Actor
    elif typ == "message":
        for ind, row in df.iterrows():
            if support == "sup":
                return_list = return_list + list(row['Actor_Support'].split(", "))
            elif support == "opp":
                return_list = return_list + list(row['Actor_Opposition'].split(", "))
            else:
                return_list = return_list + list(row['Actor_Support'].split(", ")) + \
                              list(row['Actor_Opposition'].split(", "))

    return_list = list(set(return_list))
    if "" in return_list:
        return_list.remove("")
    return return_list


def option_country(df, support="all"):
    """

    param df:
    :param support:
    :return:
        generates an option that can be used with the dropdown for actors
    """
    lst_countries = unique_actors(df, support="all")
    options_lst = []
    for country in lst_countries:
        pretty_country = country.replace("_", " ")
        dic = {'label': pretty_country, 'value': country}
        options_lst.append(dic)
    return options_lst


def unique_keywords(df) -> typing.List:
    """
    return a list with all unique keywords in the df
    :param df:
    :return:
    """
    return_list = []
    for ind, row in df.iterrows():
        if row['Venturini_keywords'] != "":
            return_list = return_list + list(row['Venturini_keywords'].split(", "))
    return list(set(return_list))


def option_keywords(df):
    """
    :param df:
    :return:
        generates an option list that can be used with the keyword dropdown.
    """
    lst_keywords = unique_keywords(df)
    options_lst = []
    for keyword in lst_keywords:
        dic = {'label': keyword, 'value': keyword}
        options_lst.append(dic)
    return options_lst


def upperpart_layout(df: pd.DataFrame, page):
    """
    :param df:
    :return:
        return the html upperpart of the layout where the user can narrow the dataframe
        in terms of actor, message, keyword
    """
    upper_layout = \
        html.Div(children=[
            html.Div([
                # html.Pre(children="Time rangeslider", style={"fontSize": "150%"}),
                dcc.RangeSlider(
                    # marks={i: '{}'.format(i) for i in range(1995, 2019)},
                    marks={str(year): str(year) for year in df['Year'].unique()},
                    min=df['Year'].min(),
                    max=df['Year'].max(),
                    step=None,
                    value=[1995, 2019],
                    id="range-slider"
                ),
            ]),
            html.Div(className='row', children=[
                html.Div(className='four columns', children=[
                    html.Div([
                        html.Pre(children="Actors :", style={"fontSize": "150%"}),
                        dcc.Dropdown(
                            options=(option_country(df)),
                            multi=True,
                            value=None,
                            id="country_input",
                        )
                    ]),
                    html.Div([
                        html.Pre(children="Search in message: ", style={"fontSize": "150%"}),
                        dcc.Input(
                            type="text",
                            value="",
                            id="search_input",
                            placeholder="Filter messages using one or more words",
                        )
                    ]),
                    html.Div([
                        html.Pre(children="Clusters : ", style={"fontSize": "150%"}),
                        dcc.Dropdown(
                            options=option_keywords(df),
                            multi=True,
                            value="",
                            id="keyword_input"
                        )
                    ])
                ]),
                html.Div(className='four columns', children=[
                    dcc.Markdown(update_markdown(df), id=page)
                ])
            ])
        ])
    return upper_layout


def generate_table(dataframe, max_rows=1000):
    """

    :param dataframe:
    :param max_rows:
    :return:
        return a Table representation of the dataframe
    """
    return html.Table(style={'border': '1px solid black', 'width': '100%', 'table-layout': 'fixed'},
                      # Header
                      children=[html.Tr([html.Th(col) for col in dataframe.columns], )] +
                               # Body
                               [html.Tr(
                                   [html.Td(dataframe.iloc[i][col], style={'font_size': '10'}) for col in
                                    dataframe.columns],

                               )
                                   for i in range(min(len(dataframe), max_rows))],

                      )


# style = {'font_size': '12px', 'width': '100%', 'overflow-x': 'auto', 'border-spacing': '0', 'display': 'block',
#          'cellspacing': '0'},
def network_graph(df, total_actors):
    """

    :param total_actors:
    :param df:
    :return:
    """
    df_actors = pd.DataFrame(unique_actors(df), columns=["Actor"])
    # removing pronouns
    cond_pronoun = df_actors["Actor"].isin(["He", "She", "They", "It"])
    df_actors = df_actors[~cond_pronoun]
    if len(df_actors) < 1:
        return go.Figure()

    total_mention_sup = Counter()
    total_mention_opp = Counter()

    for ind, row in df.iterrows():
        lst_actor_sup = sorted(row.Actor_Support.split(", "))
        lst_actor_opp = sorted(row.Actor_Opposition.split(", "))
        for mention in lst_actor_opp:
            total_mention_opp[mention] += 1
        for mention in lst_actor_sup:
            total_mention_sup[mention] += 1
    df_actors['total_sup'] = df_actors['Actor'].apply(lambda x: total_mention_sup[x])
    df_actors['total_opp'] = df_actors['Actor'].apply(lambda x: total_mention_opp[x])

    total_edge_agree = Counter()
    total_edge_disagree = Counter()

    for ind, row in df.iterrows():
        lst_actor_sup = sorted(row.Actor_Support.split(", "))
        lst_actor_opp = sorted(row.Actor_Opposition.split(", "))
        for agree in list(combinations(lst_actor_sup, 2)) + list(combinations(lst_actor_opp, 2)):
            total_edge_agree[agree] += 1
        for disagree in product(lst_actor_sup, lst_actor_opp):
            total_edge_disagree[disagree] += 1

    df_edge = pd.DataFrame.from_dict(total_edge_agree, orient="index").reset_index()
    df_edge = df_edge.rename(columns={'index': 'edge', 0: 'edge_agree'})
    df_edge['edge_disagree'] = df_edge.edge.apply(lambda x: total_edge_disagree[x])
    df_edge["c1"] = df_edge.edge.apply(lambda x: x[0])
    df_edge["c2"] = df_edge.edge.apply(lambda x: x[1])
    df_edge = df_edge.drop(columns="edge")

    df_actors = df_actors.nlargest(total_actors, "total_sup").reset_index(drop=True)
    cond_c1 = df_edge["c1"].isin(df_actors.Actor.values.tolist())
    cond_c2 = df_edge["c2"].isin(df_actors.Actor.values.tolist())
    df_edge = df_edge[cond_c1 & cond_c2]

    # Networkx
    g = nx.Graph()
    nodesize = []
    for ind, row in df_actors.iterrows():
        g.add_node(row["Actor"])
        nodesize.append(row.total_sup)
    maxi_node = max(nodesize)
    node_size = [100 * node / maxi_node for node in nodesize]
    for ind, row in df_edge.iterrows():
        c1 = row["c1"]
        c2 = row["c2"]
        count_agree = row["edge_agree"]
        count_disagree = row["edge_disagree"]
        g.add_edge(c1, c2, weight=count_agree + count_disagree, agree=row["edge_agree"], disagree=row["edge_disagree"])

    pos = nx.spring_layout(g, k=1, iterations=200)
    # pos = nx.nx_pydot.graphviz_layout(g)

    # nx.draw_networkx_nodes(g, pos, node_size=node_size)
    # labels = nx.get_edge_attributes(g, "weight")

    edges = g.edges()

    w = [g[u][v]['weight'] for u, v in edges]
    maxi = max(w) if len(w) > 0 else 0

    for node in g.nodes:
        g.nodes[node]['pos'] = list(pos[node])

    # Middle point for hovering
    middle_hover_trace = go.Scatter(x=[], y=[], hovertext=[], mode='markers', hoverinfo="text",
                                    marker={'size': 20, 'color': 'LightSkyBlue'}, opacity=0)

    # Edges = Lines logic
    edge_trace = []
    for ind, edge in enumerate(g.edges()):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = 4 * (g[edge[0]][edge[1]]["weight"]) / maxi

        # Color of the edge, blue or red or mix depending on support/opposition
        red = g[edge[0]][edge[1]]["disagree"]
        blue = g[edge[0]][edge[1]]["agree"]
        red_blue = red + blue
        red_color = red * 255 / red_blue
        blue_color = blue * 255 * red_blue
        # Line logic
        trace = go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            line=dict(width=weight, color='rgb(' + str(5 * red_color) + ',0,' + str(blue_color) + ')'),
            # hovertext='Support = ' + str(blue) + '\nOpposition = ' + str(red),
            mode='lines')
        edge_trace.append(trace)

        # Middle-point logic
        hovertext = str(edge) + ' :<br>Support = ' + str(blue) + '<br>Opposition = ' + str(red)
        middle_hover_trace['x'] += tuple([(x0 + x1) / 2])
        middle_hover_trace['y'] += tuple([(y0 + y1) / 2])
        middle_hover_trace['hovertext'] += tuple([hovertext])

    # Nodes logic
    node_x = []
    node_y = []
    node_name = []
    for node in g.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_name.append(str(node))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hovertext=nodesize,
        # hoverinfo='',
        text=node_name,
        marker=dict(
            showscale=True,
            # colorscale options
            # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=node_size,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(g.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append('# of connections: ' + str(len(adjacencies[1])))

    node_trace.marker.color = node_adjacencies
    # node_trace.text = node_text
    edge_trace.append(node_trace)
    edge_trace.append(middle_hover_trace)
    fig = go.Figure(data=edge_trace,
                    layout=go.Layout(
                        title='<br>Network graph showing actor collaboration',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text=str(len(df)) + "</a>",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    fig.update_traces(textposition='top center', textfont=dict(family='sans-serif', size=15, color='#000'))
    return fig


def update_markdown(df):
    nb_actors = len(unique_actors(df))
    nb_message = len(df)
    min_year = df.Year.min()
    max_year = df.Year.max()
    message = ""
    if len(df) > 0:
        res1 = '- Total unique actors : {}.  \n'.format(nb_actors)
        res2 = '- Total messages : {}.  \n'.format(nb_message)
        res3 = '- Messages go from {} to {}.  \n'.format(min_year, max_year)
        message = res1 + res2 + res3
    if len(df) <= 0:
        message = 'Database is empty.'

    return message
