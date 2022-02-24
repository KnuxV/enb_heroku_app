from dash import dcc, html
from dash.dependencies import Input, Output

# Connect to main app.py file
from app import app
from app import server

# Connect to your app pages
from apps import graph, table


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([
        dcc.Link('Graphs |', href='/apps/graph'),
        dcc.Link('| Table', href='/apps/table'),
    ], className="row"),
    html.Div(id='page-content', children=[])
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/apps/graph':
        return graph.layout
    if pathname == '/apps/table':
        return table.layout
    else:
        # return "404-Page Error! Please choose a link"
        return table.layout


if __name__ == '__main__':
    app.run_server(debug=True)
