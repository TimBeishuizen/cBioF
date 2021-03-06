# -*- coding: utf-8 -*-

"""This file is part of the TPOT library.

TPOT was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Daniel Angell (dpa34@drexel.edu)
    - and many more generous open source contributors

TPOT is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version.

TPOT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with TPOT. If not, see <http://www.gnu.org/licenses/>.

"""

# All feature selection tools
fs_config = ['TPOT_RFE',
             'TPOT_SelectFromModel',
             'TPOT_SelectFwe',
             'TPOT_SelectPercentile',
             'TPOT_VarianceThreshold',
             'TPOT_SelectKBest',
             'TPOT_SelectKFromModel',
             'TPOT_ForwardSelector',
             'TPOT_PTA',
             'TPOT_FloatingSelector',
             ]

order_config = ['TPOT_FeatureOrderer']

fs_pipeline_steps = ['rfe',
             'selectfrommodel',
             'selectfwe',
             'selectpercentile',
             'variancethreshold',
             'selectkbest',
             'selectkfrommodel',
             'forwardselector',
             'pta',
             'floatingselector',
             ]