# import
from re import sub
from dash import Dash, dash_table
import pandas as pd
import sqlite3
import datetime
import numpy as np
from datetime import datetime
from datetime import date
import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyOAuth
import json
import math
import os
import sys
from flask import Flask, url_for, session, request, redirect
from wordcloud import WordCloud

import matplotlib.pyplot as plt

# dash
import dash
from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px

# functions


def sql_query_pd(q):
    '''
    query the database
    retrun pandas df

    input: sql query
    '''

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    df = pd.read_sql(q, con)

    con.commit()
    con.close()

    return df


def check_genre(a, b):
    '''#input is:
    a, attributed genres - List
    b, selected genres

    see if there is common overlap between attributed genres and selected genres'''

    a_set = set(a)
    b_set = set(b)
    if a == 'n/a':
        return False

    elif len(a_set & b_set) > 0:

        return True

    else:
        return False


def load_genres(x):
    '''
    load genres into list from string
    if no genre return 'n/a

    input: unloaded genre'''

    if x != '[]':
        return json.loads(x)
    else:
        return ['n/a']


# loading df and preprocessing

db_path = 'data/music_recco.db'

df = sql_query_pd('''
SELECT tracks.*, show_individual.*, show_follow_list.show_name
FROM tracks
LEFT JOIN show_individual ON tracks.url = show_individual.url
LEFT JOIN show_follow_list ON show_individual.initial = show_follow_list.initial
WHERE NOT tracks.spotify_uri = 'n/a'
''')

# features to be used in histogram
features = ['speechiness', 'acousticness', 'instrumentalness',
            'liveness', 'loudness', 'popularity', 'tempo', 'release_date']

# drop entry with missing energy value
df = df[~df.energy.isnull()]

# drop entry with 0 tempo value
df = df[df.tempo != 0.0]

# load genres as list
df['genres_loaded'] = df['genres'].apply(load_genres)

# get list of distinct genres
distinct_genres = set(df.genres_loaded.sum())

# load release date as datetime object
df.release_date = df.release_date.apply(lambda x: datetime.fromisoformat(x))

# create year column
df['release_year'] = df.release_date.apply(lambda x: x.year)

# convert show date to date object
df.show_date2 = df.show_date2.apply(lambda x: datetime.fromisoformat(x))

# convert duration from ms to %M:%S
df['duration'] = df.duration_ms.apply(
    lambda x: datetime.fromtimestamp(x/1000.0).strftime('%M:%S'))

# get min/max release date
min_release_date = df.release_date.min().year

min_release_date = math.floor(min_release_date/10)*10

max_release_date = df.release_date.max().year

max_release_date = math.ceil(max_release_date/10)*10


# spotify login

def create_spotify_oauth():

    scope = 'playlist-modify-private user-read-private playlist-modify-public'
    # spotipy.Spotify(auth_manager=
    return SpotifyOAuth(
        scope=scope,
        # client_id=os.environ.get("CLIENT_ID"),
        # client_secret=os.environ.get("CLIENT_SECRET"),
        #redirect_uri=url_for('authorize', _external=True )
    )


# create dash app


app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

server = app.server


app.title = "Music Dashboard"


# the style arguments for the sidebar.
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "24rem",
    "padding": "2rem 2rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar
CONTENT_STYLE = {
    "margin-left": "25rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}


controls = html.Div(
    [
        html.Div(
            [
                dbc.Label("DJ/Record Label"),
                dcc.Dropdown(
                    id='radio_show',
                    options=[{'label': i.title().replace('_', ' '), 'value': i} for i in df.show_name.unique(
                    )]+[{'label': 'Select All', 'value': 'all_values'}],
                    value=['all_values'],
                    multi=True,
                    searchable=True
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("Show Date"),
                dcc.DatePickerRange(
                    id='show-date-range',
                    min_date_allowed=date(2011, 1, 1),
                    max_date_allowed=df.show_date2.max(),
                    initial_visible_month=df.show_date2.max(),
                    start_date=df.show_date2.min(),
                    end_date=df.show_date2.max()
                )
            ],
            style={'padding-top': '10px'}
        ),
        html.Div(
            [
                dbc.Label("Genre"),
                dcc.Dropdown(
                    id='genre',
                    options=[{'label': i.title(), 'value': i} for i in distinct_genres] +
                    [{'label': 'Select All', 'value': 'all_values'}],
                    value=['all_values'],
                    multi=True,
                    searchable=True
                ),
            ],
            style={'padding-top': '10px'}
        ),
        # Energy
        html.Div([
            html.Label('Energy'),
            html.Div([
                dcc.RangeSlider(
                    id='energy-range-slider',
                    min=0,
                    max=1,
                    step=0.05,
                    marks={x/10: str(x/10) for x in range(0, 11, 2)},
                    value=[0, 1],
                    allowCross=False
                )
            ],
                style={'padding-top': '5px'}
            )
        ],
            style={'padding-top': '10px'}
        ),

        # Release Year
        html.Div([
            html.Label('Release Year'),
            html.Div([
                dcc.RangeSlider(
                    id='release-date-slider',
                    min=min_release_date,
                    max=max_release_date,
                    step=10,
                    value=[min_release_date, max_release_date],
                    marks={i: str(i) for i in range(min_release_date, max_release_date+10, 10)})
            ],
                style={'padding-top': '5px'})
        ],
            style={'padding-top': '10px'}
        ),

        # create spotify playlist
        html.Div([
            html.Div(
                dcc.Input(id='playlist-input-box',
                          type='text'),
                style={'padding-bottom': '5px'}),

            html.Button('Create Spotify Playlist',
                        id='playlist-button',
                        disabled=True),
            html.Div(id='output-container-button',
                     children='Coming soon')
        ],
            style={'padding-top': '30px'}),


    ],
    # body=True,
)


sidebar = html.Div(
    [
        html.H2("Music Reccomender", className="display-4"),
        html.Hr(),
        html.P(
            "Apply your filters", className="lead"
        ),
        dbc.Nav(
            [
                controls
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)


content = html.Div(id="page-content",
                   children=[
                       dbc.Container([

                           # first row
                           dbc.Row([
                               # highlight numbers
                               dbc.Col([

                                   html.H1(id='num-of-tracks'),
                                   html.H1(id="num-of-artists"),
                                   html.H1(id='total-duration')
                               ],
                                   align='end'),

                               # wordcloud
                               dbc.Col([
                                   html.Img(id='wordcloud',
                                            style={'width': '100%'})
                               ],
                                   width=8
                               )
                           ],
                               style={'padding-bottom': '20px'}),

                           # second row - border line
                           dbc.Row([], className='divBorder'),

                           # third row
                           dbc.Row([
                               dbc.Col([],
                                       width=7
                                       ),
                               dbc.Col([html.H6('Feature Selection:')],
                                       width=2
                                       ),
                               dbc.Col([

                                   dcc.Dropdown(
                                       id='distribution-selection',
                                       options=[{'label': i.title().replace(
                                           '_', ' '), 'value': i} for i in features],
                                       value='tempo',
                                       multi=False,
                                       searchable=True)

                               ],
                                   width={"size": 3, 'order': 'last'},
                               ),
                           ],
                               align='end',
                           ),
                           # fourth row - histogram
                           dbc.Row([
                               dbc.Col([
                                   # distribution
                                   dcc.Graph(id='graph-2',
                                             config={
                                                 'displayModeBar': False
                                             },
                                             ),
                               ])
                           ],
                           ),



                           dbc.Row([

                               # table
                               html.Div([
                                   html.H2('Sample of Selected Tracks'),
                                   #html.H4('Random sample of 20 tracks from your selection'),
                                   dash_table.DataTable(
                                       id='select-tracks',
                                       cell_selectable=False,
                                       columns=[{"name": x, "id": i} for i, x in zip(['track', 'artist', 'duration', 'release_year'],
                                                                                     ['Track', 'Artist', 'Duration', 'Year'])
                                                ],
                                       style_cell={
                                        'font-family': 'sans-serif'},
                                       style_data={
                                           'textAlign': 'left', 'font': 'calibri'},
                                       style_header={'textAlign': 'left',  # this isn't working??
                                                     'fontWeight': 'bold'},
                                       style_as_list_view=True,
                                       style_data_conditional=[
                                           {
                                               'if': {'row_index': 'odd'},
                                               'backgroundColor': "#f8f9fa",
                                           }
                                       ]
                                   )
                               ]),
                           ],
                           ),

                           dcc.Store(id='selected-uris', data=[])

                       ]),
                   ],

                   style=CONTENT_STYLE)

app.layout = html.Div([sidebar, content])


@app.callback(
    [
        Output("graph-2", "figure"),
        Output("select-tracks", "data"),
        Output("selected-uris", "data"),
        Output("num-of-tracks", 'children'),
        Output("wordcloud", 'src'),
        Output("num-of-artists", "children"),
        Output("total-duration", "children"),
    ],
    [Input('radio_show', 'value'),
     Input('genre', 'value'),
     Input('energy-range-slider', 'value'),
     Input('distribution-selection', 'value'),
     Input('release-date-slider', 'value'),
     Input('show-date-range', 'start_date'),
     Input('show-date-range', 'end_date'),
     ])
def update_dashboard(radio_show,
                     genres,
                     energy_range,
                     distribution_selection,
                     release_date_range,
                     show_date_start,
                     show_date_end):

    # masks
    energy_mask = (df.energy.between(energy_range[0], energy_range[1]))

    release_date_mask = (df.release_date.between((datetime.strptime(str(release_date_range[0]), '%Y')),
                                                 (datetime.strptime(str(release_date_range[1]-1), '%Y'))))

    radio_show_mask = (df.show_name.isin(radio_show))

    genre_mask = (df.genres_loaded.apply(check_genre, b=genres))

    show_date_mask = (df.show_date2.between(show_date_start, show_date_end))

    # to add select all option in Dropdowns
    inputs = [radio_show, genres]
    masks = [radio_show_mask, genre_mask]

    masks_output = [m for i, m in zip(inputs, masks) if 'all_values' not in i]

    # adding masks that don't require a 'select all' option
    masks_output += [
        show_date_mask,
        release_date_mask,
        energy_mask]

    if show_date_start == None or show_date_end == None:

        masks_output.remove(show_date_mask)

    # create new df applying all masks

    df_mask = df[np.logical_and.reduce(masks_output)]

    # remove duplicates (songs played on multiple different shows)
    # filter the mask

    # highlight stats
    #total_num = df.shape[0]
    selected_num = df_mask.spotify_uri.unique().shape[0]
    #remaining = total_num-selected_num

    def num2str(n, text):
        '''
        convert num 2 string

        if over 1000 retur 1000+
        '''
        if n > 1000:
            output = '1000+'
        else:
            output = str(n)

        return f'{output} {text}'

    # num of tracks

    num_of_tracks = num2str(selected_num, 'tracks')

    # num of artists

    num_of_artists = num2str(df_mask.spotify_artist.unique().shape[0],
                             'artists')

    # total duration of selected tracks

    total_duration = pd.to_timedelta(
        df_mask.duration_ms, unit='ms').sum().components

    if total_duration.days > 0:
        try:
            output = total_duration.days+(total_duration.hours/24)
        except ZeroDivisionError:
            output = total_duration.days

        output_text = str(round(output, 1))+' '+'day' + \
            ('s' if output > 1 else '')

    elif total_duration.hours > 0:
        try:
            output = total_duration.hours+(total_duration.minutes/60)
        except ZeroDivisionError:
            output = total_duration.hours

        output_text = str(round(output, 1))+' '+'hour' + \
            ('s' if output > 1 else '')

    elif total_duration.minutes >= 0:

        output_text = str(round(total_duration.minutes, 0))+' ' + \
            'minute'+('' if total_duration.minutes == 1 else 's')

    # wordcloud

    # get a list of all the genres
    genres = df_mask[df_mask.genres != '[]'].genres_loaded.sum()

    # create a frequency dictionary
    freq_dict = {}
    if selected_num > 1:
        for genre in genres:
            try:
                freq_dict[genre] += 1
            except KeyError:
                freq_dict[genre] = 1
    else:
        # incase 0 tracks are selected
        freq_dict = {'nada': 3, 'zilch': 10, 'zero': 20,
                     'nuffin': 3, 'none': 17, 'nil': 2}

    # create word cloud
    wc = WordCloud(max_words=100,
                   width=1600,
                   height=800,
                   background_color='white').generate_from_frequencies(freq_dict)

    # Histogram

    fig2 = px.histogram(
        df_mask,
        x=distribution_selection,
        template='simple_white',
        color_discrete_sequence=['#a3a4bd'],
    )

    fig2.update_layout(
        yaxis_title_text='Count',
        xaxis_title_text=distribution_selection.title().replace('_', ' '),
        margin=dict(t=0)
    )

    # data table
    # to fix issue with table refreshing everytime a playlist title is entered
    # set the random set (could set it to a random number changing every day??)

    # selected_tracks table
    track_limit = 20

    if df_mask.shape[0] > track_limit:
        # random sample if selection is large than limit
        selected_tracks = df_mask.sample(track_limit, random_state=1)

    else:
        # can't use .sample if selections < limit so use .head
        selected_tracks = df_mask.head(track_limit)

    # drop duplicat tracks
    selected_tracks = selected_tracks.drop_duplicates(subset=['spotify_uri'])

    selected_tracks_table = selected_tracks[[
        'track', 'artist', 'spotify_album', 'duration', 'release_year']].to_dict('records')

    # return all outputs
    return fig2, selected_tracks_table, selected_tracks.spotify_uri, num_of_tracks, wc.to_image(), num_of_artists, output_text


if __name__ == '__main__':
    app.run_server(debug=True)
