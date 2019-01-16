import sys
import ast
import numpy as np
import pandas as pd
import pyproj
from pysheds.grid import Grid
try:
    import scipy.signal
    _HAS_SCIPY = True
except:
    _HAS_SCIPY = False

class SwmmIngester(object):
    """
    Container class for writing SWMM output files
    """
    def __init__(self, grid_instance, dem, fdir, catch, acc, projection, ter=None, cover=None,
                 threshold=100, control=False, initialize=False, node_depths=None,
                 link_flows=None, outlet=472, into_outlet=644):
        self.covermap = {
            11 : 0.001,
            21 : 0.0404,
            22 : 0.0678,
            23 : 0.0678,
            24 : 0.0404,
            31 : 0.0113,
            41 : 0.36,
            42 : 0.32,
            52 : 0.40,
            71 : 0.368,
            81 : 0.325,
            82 : 0.037
        }
        self.grid = grid_instance
        self.dem = dem
        self.fdir = fdir
        self.catch = catch
        self.acc = acc
        self.projection = projection
        self.ter = ter
        self.cover = cover
        self.control = control
        self.initialize = initialize
        self.in_catch = np.where(grid_instance.mask.ravel())
        self.mask = self.grid.acc.ravel() >= threshold
        self.outlet = outlet
        self.into_outlet = into_outlet
        self.dirmap = getattr(grid_instance, catch).metadata['dirmap']
        self.flat_idx = np.arange(self.grid.size)
        self.startnodes, self.endnodes = grid_instance._construct_matching(
                                         grid_instance.view(catch, dtype=np.int64),
                                         self.flat_idx.astype(np.int64),
                                         self.dirmap)
        self.start = self.startnodes[~self.mask]
        self.end = self.endnodes[~self.mask]
        A = scipy.sparse.lil_matrix((self.grid.size, self.grid.size))
        for i,j in zip(self.startnodes, self.endnodes):
            A[i,j] = 1
            A[j,i] = 1
        A = A.tocsr()
        sp = scipy.sparse.csgraph.shortest_path(A, indices=[self.outlet])
        C = scipy.sparse.lil_matrix((self.grid.size, self.grid.size))
        for i,j in zip(self.start, self.end):
            C[i,j] = 1
            C[j,i] = 1
        C = C.tocsr()
        n, labels = scipy.sparse.csgraph.connected_components(C)
        s = pd.Series(labels)
        c = pd.Series(np.bincount(labels))
        sc = s.map(c)
        self.nodes = np.unique(np.concatenate([self.startnodes[self.mask],
                                               self.endnodes[self.mask]]))
        if node_depths is not None:
            self.node_depths = node_depths
        else:
            self.node_depths = pd.Series(np.zeros(len(self.nodes)), index=self.nodes)
        if link_flows is not None:
            self.link_flows = link_flows
        # Add else case later
        self.dists = self.grid.cell_distances('catch', as_crs=projection, inplace=False)
        self.areas = self.grid.cell_area(inplace=False, as_crs=projection)
        self.slopes = self.grid.cell_slopes(catch, dem, dirmap=self.dirmap,
                                            as_crs=projection, inplace=False)
        self.slopes.flat[into_outlet] = 0.01
        self.slopes.flat[outlet] = 0.01
        d = {}
        for node in self.nodes:
            d[node] = {}
            label = labels[node]
            attached = np.where(labels == label)[0]
            attached_area = self.areas.flat[attached]
            attached_slopes = self.slopes.flat[attached]
            attached_slopes[attached_slopes == 0] = 0.0001
            attached_dists = sp.flat[attached]
            if self.ter is not None:
                attached_imperv = self.ter.flat[attached]
                avg_imperv = np.nanmean(attached_imperv)
                d[node]['pct_imperv'] = avg_imperv
            if self.cover is not None:
                attached_cover = self.cover.flat[attached]
                avg_n = pd.Series(attached_cover).map(self.covermap).mean()
                d[node]['avg_n'] = avg_n
            longest_path_dist = attached_dists.max() - attached_dists.min()
            avg_cell_dist = self.dists.flat[attached].mean()
            flow_len = 1 + longest_path_dist * avg_cell_dist
            total_area = attached_area.sum()
            avg_slope = attached_slopes.mean()
            width = total_area / flow_len
            d[node]['total_area'] = np.asscalar(1e-4*total_area)
            d[node]['width'] = np.asscalar(width)
            d[node]['pct_slope'] = np.asscalar(100*avg_slope)
        self.total_areas = pd.Series((d[node]['total_area'] for node in self.nodes),
                                     index=self.nodes)
        self.widths = pd.Series((d[node]['width'] for node in self.nodes),
                                index=self.nodes)
        self.pct_slopes = pd.Series((d[node]['pct_slope'] for node in self.nodes),
                                    index=self.nodes)
        if self.ter is not None:
            self.pct_imperv = pd.Series((d[node]['pct_imperv'] for node in self.nodes),
                                        index=self.nodes)
        else:
            self.pct_imperv = pd.Series(np.repeat(36, len(self.nodes)))
        if self.cover is not None:
            self.avg_n = pd.Series((d[node]['avg_n'] for node in self.nodes),
                                        index=self.nodes)
        else:
            self.avg_n = pd.Series(np.repeat(0.1, len(self.nodes)))
        self.startnodes = self.startnodes[self.mask]
        self.endnodes = self.endnodes[self.mask]
        non_outlet = self.startnodes != outlet
        self.startnodes = self.startnodes[non_outlet]
        self.endnodes = self.endnodes[non_outlet]

    def generate_title(self, **kwargs):
        title = 'Test Watershed'
        self.title = title

    def generate_options(self, **kwargs):
        options = {
            "FLOW_UNITS" : "CMS",
            "INFILTRATION" : "GREEN_AMPT",
            "FLOW_ROUTING" : "DYNWAVE",
            "START_DATE" : "07/05/2014",
            "START_TIME" : "00:00:00",
            "REPORT_START_DATE" : "07/05/2014",
            "REPORT_START_TIME" : "00:00:00",
            "END_DATE" : "07/07/2014",
            "END_TIME" : "00:00:00",
            "SWEEP_START" : "01/01",
            "SWEEP_END" : "12/31",
            "DRY_DAYS" : "5.000000",
            "REPORT_STEP" : "00:05:00",
            "WET_STEP" : "00:01:00",
            "DRY_STEP" : "00:05:00",
            "ROUTING_STEP" : "5",
            "ALLOW_PONDING" : "YES",
            "INERTIAL_DAMPING" : "PARTIAL",
            "VARIABLE_STEP" : "0.75",
            "LENGTHENING_STEP" : "0",
            "MIN_SURFAREA" : "1.167",
            "NORMAL_FLOW_LIMITED" : "BOTH",
            "SKIP_STEADY_STATE" : "NO",
            "FORCE_MAIN_EQUATION" : "H-W",
            "LINK_OFFSETS" : "DEPTH",
            "MIN_SLOPE" : "0",
            "IGNORE_SNOWMELT" : "YES",
            "IGNORE_GROUNDWATER" : "YES",
            "MAX_TRIALS" : "8",
            "HEAD_TOLERANCE" : "0.0015",
            "SYS_FLOW_TOL" : "5",
            "LAT_FLOW_TOL" : "5",
            "MINIMUM_STEP" : "0.5",
            "THREADS" : "2",
        }
        # Manual overrides
        for key, value in kwargs.items():
            options[key] = value
        self.options = options

    def generate_evaporation(self, **kwargs):
        evaporation = {
            "CONSTANT" : "0.000000",
            "DRY_ONLY" : "NO"
        }
        # Manual overrides
        for key, value in kwargs.items():
            evaporation[key] = value
        self.evaporation = evaporation

    def generate_report(self, **kwargs):
        report = {
            "INPUT" : "NO",
            "CONTROLS" : "NO",
            "SUBCATCHMENTS" : "ALL",
            "NODES" : "ALL",
            "LINKS"	: "ALL",
        }
        # Manual overrides
        for key, value in kwargs.items():
            report[key] = value
        self.report = report

    def generate_raingages(self, **kwargs):
        raingages = {}
        raingages['name'] = 'R0'
        raingages['rain_type'] = 'VOLUME'
        raingages['time_interval'] = 0.083333
        raingages['snow_catch'] = 1.0
        raingages['data_source'] = 'TIMESERIES STEP_INPUT'
        raingages = pd.DataFrame.from_dict(raingages, orient='index').T
        # Manual overrides
        for key, value in kwargs.items():
            raingages[key] = value
        self.raingages = raingages[['name', 'rain_type', 'time_interval',
                                    'snow_catch', 'data_source']]

    def generate_subcatchments(self, uniform_imperv=0, **kwargs):
        subcatchments = {}
        subcatchments['name'] = 'S' + pd.Series(self.nodes).astype(str)
        subcatchments['raingage'] = pd.Series(np.repeat('R0', len(self.nodes)))
        subcatchments['outlet'] = 'J' + pd.Series(self.nodes).astype(str)
        # TODO: Assuming hectares
        subcatchments['total_area'] = (self.total_areas).values
        if uniform_imperv:
            subcatchments['pct_imperv'] = pd.Series(np.repeat(uniform_imperv, len(self.nodes)))
        else:
            subcatchments['pct_imperv'] = self.pct_imperv.values
        subcatchments['width'] = self.widths.values
        subcatchments['pct_slope'] = self.pct_slopes.values
        subcatchments['curb_length'] = pd.Series(np.repeat(0, len(self.nodes)))
        subcatchments['snow_pack'] = pd.Series(np.repeat('', len(self.nodes)))
        subcatchments = pd.DataFrame.from_dict(subcatchments)
        # Manual fixes for outlet
        # TODO: Generalize
        # subcatchments.loc[subcatchments['name'] == 'S{0}'.format(self.outlet), 'width'] = 60
        # subcatchments.loc[subcatchments['name'] == 'S{0}'.format(self.outlet), 'pct_slope'] = 0
        # subcatchments.loc[subcatchments['name'] == 'S{0}'.format(self.into_outlet), 'width'] = 60
        # subcatchments.loc[subcatchments['name'] == 'S{0}'.format(self.into_outlet), 'pct_slope'] = 0
        # Manual overrides
        for key, value in kwargs.items():
            subcatchments[key] = value
        self.subcatchments = subcatchments[['name', 'raingage', 'outlet', 'total_area',
                                            'pct_imperv', 'width', 'pct_slope',
                                            'curb_length', 'snow_pack']]

    def generate_subareas(self, uniform_n=0, **kwargs):
        subareas = {}
        subareas['subcatchment'] = 'S' + pd.Series(self.nodes).astype(str)
        subareas['n_imperv'] = pd.Series(np.repeat(0.01, len(self.nodes)))
        if uniform_n:
            subareas['n_perv'] = pd.Series(np.repeat(uniform_n, len(self.nodes)))
        else:
            subareas['n_perv'] = self.avg_n.fillna(0.1).values
        subareas['s_imperv'] = pd.Series(np.repeat(0.05, len(self.nodes)))
        subareas['s_perv'] = pd.Series(np.repeat(0.05, len(self.nodes)))
        subareas['pct_zero'] = pd.Series(np.repeat(25, len(self.nodes)))
        subareas['route_to'] = pd.Series(np.repeat('OUTLET', len(self.nodes)))
        subareas['pct_routed'] = pd.Series(np.repeat('', len(self.nodes)))
        subareas = pd.DataFrame.from_dict(subareas)
        # Manual overrides
        for key, value in kwargs.items():
            subareas[key] = value
        self.subareas = subareas[['subcatchment', 'n_imperv', 'n_perv', 's_imperv', 's_perv',
                                  'pct_zero', 'route_to', 'pct_routed']]

    def generate_infiltration(self, **kwargs):
        infiltration = {}
        infiltration['subcatchment'] = 'S' + pd.Series(self.nodes).astype(str)
        infiltration['suction'] = pd.Series(np.repeat(100.0, len(self.nodes)))
        infiltration['hyd_con'] = pd.Series(np.repeat(20.0, len(self.nodes)))
        infiltration['imd_max'] = pd.Series(np.repeat(0.2, len(self.nodes)))
        infiltration = pd.DataFrame.from_dict(infiltration)
        # Manual overrides
        for key, value in kwargs.items():
            infiltration[key] = value
        self.infiltration = infiltration[['subcatchment', 'suction', 'hyd_con', 'imd_max']]

    def generate_junctions(self, **kwargs):
        junctions = {}
        junctions['name'] = 'J' + pd.Series(self.nodes).astype(str)
        junctions['elevation'] = pd.Series(self.grid.view(self.dem).flat[self.nodes])
        junctions['maxdepth'] = pd.Series(np.repeat(10, len(self.nodes))) # Change this if errors
        if self.initialize:
            junctions['initdepth'] = junctions['name'].map(self.node_depths)
        else:
            junctions['initdepth'] = pd.Series(np.repeat(0, len(self.nodes)))
        junctions['surdepth'] = pd.Series(np.repeat(0, len(self.nodes)))
        junctions['aponded'] = pd.Series(np.repeat(0, len(self.nodes)))
        junctions = pd.DataFrame.from_dict(junctions)
        # Manual fixes
        junctions.loc[junctions['name'] == 'J{0}'.format(self.outlet), 'elevation'] = 162
        # Manual overrides
        for key, value in kwargs.items():
            junctions[key] = value
        self.junctions = junctions[['name', 'elevation', 'maxdepth', 'initdepth',
                                    'surdepth', 'aponded']]

    def generate_conduits(self, **kwargs):
        conduits = {}
        conduits['name'] = ('J' + pd.Series(self.startnodes).astype(str) + '_'
                            + 'J' + pd.Series(self.endnodes).astype(str))
        conduits['inlet_node'] = 'J' + pd.Series(self.startnodes).astype(str)
        conduits['outlet_node'] = 'J' + pd.Series(self.endnodes).astype(str)
        conduits['length'] = pd.Series(self.dists.flat[self.startnodes])
        conduits['mannings_n'] = pd.Series(np.repeat(0.05, len(self.startnodes)))
        conduits['inlet_offset'] = pd.Series(np.repeat(0, len(self.startnodes)))
        conduits['outlet_offset'] = pd.Series(np.repeat(0, len(self.startnodes)))
        if self.initialize:
            conduits['init_flow'] = conduits['name'].map(self.link_flows)
        else:
            conduits['init_flow'] = pd.Series(np.repeat(0, len(self.startnodes)))
        conduits['max_flow'] = pd.Series(np.repeat(0, len(self.startnodes)))
        conduits = pd.DataFrame.from_dict(conduits)
        # Manual fixes
        # TODO: Broken
        termout = pd.Series({'name' : 'TERMOUT',
                            'inlet_node' : 'J' + str(np.argmax(self.grid.acc)),
                            'outlet_node' :'TERM',
                            'length' : 1,
                            'mannings_n' : 0.05,
                            'inlet_offset' : 0,
                            'outlet_offset' : 0,
                            'init_flow' : 0,
                            'max_flow' : 0}, name=len(conduits))
        conduits = conduits.append(termout)
        # Manual overrides
        for key, value in kwargs.items():
            conduits[key] = value
        self.conduits = conduits[['name', 'inlet_node', 'outlet_node', 'length',
                                  'mannings_n', 'inlet_offset', 'outlet_offset',
                                  'init_flow', 'max_flow']]

    def generate_channel_dims(self, a=7.2, b=0.5, c=0.5, f=0.3, d_offset=1.1, w_offset=1.3,
                     d_scale=1/5000, w_scale=1/5000, **kwargs):
        # From here: http://onlinelibrary.wiley.com/doi/10.1002/wrcr.20440/full
        channel_w = self.grid.view(self.acc).copy()
        channel_d = self.grid.view(self.acc).copy()
        channel_w = (a*(w_scale * channel_w)**b) + w_offset
        channel_d = (c*(d_scale * channel_d)**f) + d_offset
        self.channel_w = channel_w
        self.channel_d = channel_d

    def generate_transects(self, lfactor=1, hfactor=1.5):
        transect_names = 'T_' + self.conduits['name'].iloc[:-1]
        nc = "NC\t0.05\t0.05\t0.05"
        x1 = "X1\t" + transect_names + "\t6\t0\t0\t0\t0\t0\t0\t0\t0"
        d = pd.Series(self.channel_d.flat[self.startnodes])
        w = pd.Series(self.channel_w.flat[self.startnodes])
        zeros = pd.Series(np.repeat(0, len(self.startnodes)))
        gr = [d + hfactor*d, zeros,
              d, lfactor*w,
              zeros, lfactor*w,
              zeros, lfactor*w + w,
              d, lfactor*w + w,
              d + hfactor*d, 2*lfactor*w + w]
        gr = 'GR\t' + pd.concat([col.astype(str) + '\t' for col in gr], axis=1).sum(axis=1)
        transects = nc + '\n' + x1 + '\n' + gr + '\n\n'
        termout = ("NC\t0.05\t0.05\t0.05" + '\n' +
                   "X1\tT_TERMOUT\t6\t0\t0\t0\t0\t0\t0\t0\t0" + '\n' +
                   gr.loc[len(gr) - 1])
        transects = transects.sum()
        transects = transects + termout
        self.transects = transects

    def generate_xsections(self, transect=True, **kwargs):
        xsections = {}
        xsections['link'] = self.conduits['name'].iloc[:-1]
        if transect:
            xsections['shape'] = pd.Series(np.repeat('IRREGULAR', len(self.startnodes)))
            xsections['geom_1'] = 'T_' + xsections['link']
            xsections['geom_2'] = pd.Series(np.repeat(0, len(self.startnodes)))
            xsections['geom_3'] = pd.Series(np.repeat(0, len(self.startnodes)))
            xsections['geom_4'] = pd.Series(np.repeat(0, len(self.startnodes)))
            xsections['barrels'] = pd.Series(np.repeat(1, len(self.startnodes)))
            xsections = pd.DataFrame.from_dict(xsections)
            termout = pd.Series({'link' : 'TERMOUT',
                                 'shape' : 'IRREGULAR',
                                 'geom_1' : 'T_TERMOUT',
                                 'geom_2' : 0,
                                 'geom_3' : 0,
                                 'geom_4' : 0,
                                 'barrels' : 1}, name=len(xsections))
            xsections = xsections.append(termout)
            # Manual overrides
            for key, value in kwargs.items():
                xsections[key] = value
            self.xsections = xsections[['link', 'shape', 'geom_1', 'geom_2',
                                        'geom_3', 'geom_4', 'barrels']]
        else:
            xsections['shape'] = pd.Series(np.repeat('RECT_OPEN', len(self.startnodes)))
            xsections['geom_1'] = pd.Series(self.channel_d.flat[self.startnodes])
            xsections['geom_2'] = pd.Series(self.channel_w.flat[self.startnodes])
            xsections['geom_3'] = pd.Series(np.repeat(0, len(self.startnodes)))
            xsections['geom_4'] = pd.Series(np.repeat(0, len(self.startnodes)))
            xsections['barrels'] = pd.Series(np.repeat(1, len(self.startnodes)))
            xsections = pd.DataFrame.from_dict(xsections)
            termout = pd.Series({'link' : 'TERMOUT',
                                'shape' : 'RECT_OPEN',
                                'geom_1' : xsections.loc[len(xsections) - 1, 'geom_1'],
                                'geom_2' : xsections.loc[len(xsections) - 1, 'geom_2'],
                                'geom_3' : 0,
                                'geom_4' : 0,
                                'barrels' : 1}, name=len(xsections))
            xsections = xsections.append(termout)
            # Manual overrides
            for key, value in kwargs.items():
                xsections[key] = value
            self.xsections = xsections[['link', 'shape', 'geom_1', 'geom_2',
                                        'geom_3', 'geom_4', 'barrels']]

    def generate_orifices(self, **kwargs):
        orifices = {}
        orifices = pd.DataFrame(columns=['name', 'node1', 'node2', 'type', 'offset', 'cd'])
        self.orifices = orifices

    def generate_outfalls(self, invert_elevation=None, **kwargs):
        if invert_elevation is None:
            invert_elevation = self.grid.view(self.dem).flat[self.into_outlet] - 1
        outfalls = {}
        outfalls['name'] = 'TERM'
        outfalls['invert_elevation'] = invert_elevation
        outfalls['outfall_type'] = 'FREE'
        outfalls['stage_table_ts'] = ''
        outfalls['tide_gate'] = 'NO'
        outfalls['route_to'] = ''
        outfalls = pd.DataFrame.from_dict(outfalls, orient='index').T
        # Manual overrides
        for key, value in kwargs.items():
            outfalls[key] = value
        self.outfalls = outfalls[['name', 'invert_elevation', 'outfall_type',
                                  'stage_table_ts', 'tide_gate', 'route_to']]

    def generate_timeseries(self, intensity=1.5, **kwargs):
        timeseries = {}
        timelen = 5
        timeseries['name'] = pd.Series(np.repeat('STEP_INPUT', timelen))
        timeseries['date'] = ''
        timeseries['time'] = pd.Series(pd.date_range('20140705', periods=timelen,
                                                     freq='5min').strftime('%H:%M'))
        timeseries['value'] = pd.Series(intensity*scipy.signal.unit_impulse(timelen))
        timeseries = pd.DataFrame.from_dict(timeseries)
        # Manual overrides
        for key, value in kwargs.items():
            timeseries[key] = value
        self.timeseries = timeseries[['name', 'date', 'time', 'value']]

    def generate_storage_uncontrolled(self, ixes, hfactor=1.5, **kwargs):
        storage_uncontrolled = {}
        depths = 4
        init_depths = 0.1
        storage_ends = [np.asscalar(self.endnodes[np.where(self.startnodes == ix)])
                        for ix in ixes]
        storage_uncontrolled['name'] = 'ST' + pd.Series(ixes).astype(str)
        storage_uncontrolled['elev'] = self.grid.view(self.dem).flat[storage_ends]
        storage_uncontrolled['ymax'] = hfactor * self.channel_d.flat[ixes] + self.channel_d.flat[ixes]
        storage_uncontrolled['y0'] = 0
        storage_uncontrolled['Acurve'] = 'FUNCTIONAL'
        storage_uncontrolled['A0'] = self.channel_w.flat[ixes]
        storage_uncontrolled['A1'] = 0
        storage_uncontrolled['A2'] = 1
        storage_uncontrolled = pd.DataFrame.from_dict(storage_uncontrolled)
        # Manual overrides
        for key, value in kwargs.items():
            storage_uncontrolled[key] = value
        self.storage_uncontrolled = storage_uncontrolled[['name', 'elev', 'ymax', 'y0', 'Acurve',
                            'A1', 'A2', 'A0']]

    def generate_storage_controlled(self, ixes, hfactor=1.5, **kwargs):
        storage_controlled = {}
        depths = 2
        init_depths = 0.1
        storage_ends = [np.asscalar(self.endnodes[np.where(self.startnodes == ix)])
                        for ix in ixes]
        storage_controlled['name'] = 'C' + pd.Series(ixes).astype(str)
        storage_controlled['elev'] = self.grid.view(self.dem).flat[storage_ends]
        storage_controlled['ymax'] = hfactor * self.channel_d.flat[ixes] + self.channel_d.flat[ixes]
        storage_controlled['y0'] = 0
        storage_controlled['Acurve'] = 'FUNCTIONAL'
        storage_controlled['A0'] = 1000
        storage_controlled['A1'] = 75000
        storage_controlled['A2'] = 1
        storage_controlled = pd.DataFrame.from_dict(storage_controlled)
        # Manual overrides
        for key, value in kwargs.items():
            storage_controlled[key] = value
        self.storage_controlled = storage_controlled[['name', 'elev', 'ymax', 'y0', 'Acurve',
                            'A1', 'A2', 'A0']]

    def generate_storage(self, **kwargs):
        storage = pd.concat([self.storage_uncontrolled, self.storage_controlled])
        storage.reset_index(drop=True, inplace=True)
        self.storage = storage

    def generate_coordinates(self, **kwargs):
        coordinates = {}
        coordinates['node'] = pd.concat([self.junctions['name'], self.storage['name']])
        coord_ix = coordinates['node'].str.extract('(\d+)').astype(int).values
        coordinates['x_coord'] = np.unravel_index(coord_ix, self.grid.shape)[1].ravel()
        coordinates['y_coord'] = np.unravel_index(coord_ix, self.grid.shape)[0].ravel()
        coordinates = pd.DataFrame.from_dict(coordinates)
        # Manual overrides
        for key, value in kwargs.items():
            coordinates[key] = value
        self.coordinates = coordinates[['node', 'x_coord', 'y_coord']]

    def generate_polygons(self, **kwargs):
        polygons = {}
        polygons['subcatchment'] = 'S' + pd.Series(self.nodes).astype(str)
        polygons['x_coord'] = np.unravel_index(self.nodes, self.grid.shape)[1]
        polygons['y_coord'] = np.unravel_index(self.nodes, self.grid.shape)[0]
        polygons = pd.DataFrame.from_dict(polygons)
        # Manual overrides
        for key, value in kwargs.items():
            polygons[key] = value
        self.polygons = polygons[['subcatchment', 'x_coord', 'y_coord']]

    def generate_map(self, **kwargs):
        mapconfig = \
        '''[MAP]
        DIMENSIONS  	0	0	{0}	{1}
        UNITS 	None
        '''.format(self.grid.shape[1], self.grid.shape[0])
        self.mapconfig = mapconfig

    def generate_control_points(self, ixes, transect=True, frac=0.1, position='top', hfactor=1.5, **kwargs):
        depths = pd.Series(self.channel_d.flat[self.startnodes], index=self.startnodes)
        widths = pd.Series(self.channel_w.flat[self.startnodes], index=self.startnodes)
        for ix in ixes:
            b = (self.conduits['inlet_node'] == ('J' + str(ix)))
            original = self.conduits.loc[b].copy()
            downstream_node = original.iloc[0]['name'].split('_')[1]
            storage_node = 'ST' + str(ix)
            storage_ctrl_node = 'C' + str(ix)
            self.conduits.loc[b, 'name'] = 'J{0}_{1}'.format(ix, storage_node)
            self.conduits.loc[b, 'outlet_node'] = storage_node
            b1 = (self.xsections['link'] == original.iloc[0]['name'])
            old_xsection = self.xsections[b1].iloc[0]
            inletnode = int(old_xsection['link'].split('_')[0].split('J')[1])
            d = float(depths.loc[inletnode])
            w = float(widths.loc[inletnode])
            self.xsections.loc[b1, 'link'] = 'J{0}_{1}'.format(ix, storage_node)
            if position == 'top':
                offset = (1 - frac)*d
            else:
                offset = 0
            if frac:
                orif = pd.Series(['{0}_{1}'.format(storage_node, downstream_node),
                                    storage_node, downstream_node, 'BOTTOM', 0, 0.5], 
                                    index=['name', 'node1', 'node2', 'type', 'offset', 'cd'],
                                    name=len(self.orifices))
                orif2 = pd.Series(['{0}_{1}'.format(storage_node, storage_ctrl_node),
                                    storage_node, storage_ctrl_node, 'BOTTOM', 0, 0.5], 
                                    index=['name', 'node1', 'node2', 'type', 'offset', 'cd'],
                                    name=len(self.orifices))
                orif3 = pd.Series(['{0}_{1}'.format(storage_ctrl_node, downstream_node),
                                    storage_ctrl_node, downstream_node, 'BOTTOM', offset, 0.5], 
                                    index=['name', 'node1', 'node2', 'type', 'offset', 'cd'],
                                    name=len(self.orifices))
            else:
                orif = pd.Series(['{0}_{1}'.format(storage_node, downstream_node),
                                    storage_node, downstream_node, 'BOTTOM', 0, 0.5], 
                                    index=['name', 'node1', 'node2', 'type', 'offset', 'cd'],
                                    name=len(self.orifices))
                orif2 = pd.Series(['{0}_{1}'.format(storage_node, storage_ctrl_node),
                                    storage_node, storage_ctrl_node, 'BOTTOM', 0, 0.5], 
                                    index=['name', 'node1', 'node2', 'type', 'offset', 'cd'],
                                    name=len(self.orifices))
                orif3 = pd.Series(['{0}_{1}'.format(storage_ctrl_node, downstream_node),
                                    storage_ctrl_node, downstream_node, 'BOTTOM', 0, 0.5], 
                                    index=['name', 'node1', 'node2', 'type', 'offset', 'cd'],
                                    name=len(self.orifices))
            self.orifices = self.orifices.append(orif)
            self.orifices = self.orifices.append(orif2)
            self.orifices = self.orifices.append(orif3)
            if transect:
                if frac:
                    xsect = pd.Series(['{0}_{1}'.format(storage_node, downstream_node),
                                        'RECT_CLOSED', d + hfactor*d, w, 0, 0],
                                        index=['link', 'shape', 'geom_1', 'geom_2', 'geom_3', 'geom_4'],
                                        name=len(self.xsections))
                    xsect2 = pd.Series(['{0}_{1}'.format(storage_node, storage_ctrl_node),
                                        'RECT_CLOSED', d + hfactor*d, w, 0, 0],
                                        index=['link', 'shape', 'geom_1', 'geom_2', 'geom_3', 'geom_4'],
                                        name=len(self.xsections))
                    xsect3 = pd.Series(['{0}_{1}'.format(storage_ctrl_node, downstream_node),
                                        'RECT_CLOSED', frac*d, w, 0, 0],
                                        index=['link', 'shape', 'geom_1', 'geom_2', 'geom_3', 'geom_4'],
                                        name=len(self.xsections))
                else:
                    xsect = pd.Series(['{0}_{1}'.format(storage_node, downstream_node),
                                        'RECT_CLOSED', d + hfactor*d, w, 0, 0],
                                        index=['link', 'shape', 'geom_1', 'geom_2', 'geom_3', 'geom_4'],
                                        name=len(self.xsections))
                    xsect2 = pd.Series(['{0}_{1}'.format(storage_node, storage_ctrl_node),
                                        'RECT_CLOSED', d + hfactor*d, w, 0, 0],
                                        index=['link', 'shape', 'geom_1', 'geom_2', 'geom_3', 'geom_4'],
                                        name=len(self.xsections))
                    xsect3 = pd.Series(['{0}_{1}'.format(storage_ctrl_node, downstream_node),
                                        'RECT_CLOSED', d, w, 0, 0],
                                        index=['link', 'shape', 'geom_1', 'geom_2', 'geom_3', 'geom_4'],
                                        name=len(self.xsections))
            else:
                if frac:
                    xsect = pd.Series(['{0}_{1}'.format(storage_node, downstream_node),
                                        'RECT_CLOSED', old_xsection['geom_1'], old_xsection['geom_2'],
                                        old_xsection['geom_3'], old_xsection['geom_4']], 
                                        index=['link', 'shape', 'geom_1', 'geom_2', 'geom_3', 'geom_4'],
                                        name=len(self.xsections))
                    xsect2 = pd.Series(['{0}_{1}'.format(storage_node, storage_ctrl_node),
                                        'RECT_CLOSED', old_xsection['geom_1'], old_xsection['geom_2'],
                                        old_xsection['geom_3'], old_xsection['geom_4']], 
                                        index=['link', 'shape', 'geom_1', 'geom_2', 'geom_3', 'geom_4'],
                                        name=len(self.xsections))
                    xsect3 = pd.Series(['{0}_{1}'.format(storage_ctrl_node, downstream_node),
                                        'RECT_CLOSED', frac*old_xsection['geom_1'], old_xsection['geom_2'],
                                        old_xsection['geom_3'], old_xsection['geom_4']], 
                                        index=['link', 'shape', 'geom_1', 'geom_2', 'geom_3', 'geom_4'],
                                        name=len(self.xsections))
                else:
                    xsect = pd.Series(['{0}_{1}'.format(storage_node, downstream_node),
                                        'RECT_CLOSED', old_xsection['geom_1'], old_xsection['geom_2'],
                                        old_xsection['geom_3'], old_xsection['geom_4']], 
                                        index=['link', 'shape', 'geom_1', 'geom_2', 'geom_3', 'geom_4'],
                                        name=len(self.xsections))
                    xsect2 = pd.Series(['{0}_{1}'.format(storage_node, storage_ctrl_node),
                                        'RECT_CLOSED', old_xsection['geom_1'], old_xsection['geom_2'],
                                        old_xsection['geom_3'], old_xsection['geom_4']], 
                                        index=['link', 'shape', 'geom_1', 'geom_2', 'geom_3', 'geom_4'],
                                        name=len(self.xsections))
                    xsect3 = pd.Series(['{0}_{1}'.format(storage_ctrl_node, downstream_node),
                                        'RECT_CLOSED', old_xsection['geom_1'], old_xsection['geom_2'],
                                        old_xsection['geom_3'], old_xsection['geom_4']], 
                                        index=['link', 'shape', 'geom_1', 'geom_2', 'geom_3', 'geom_4'],
                                        name=len(self.xsections))
            self.xsections = self.xsections.append(xsect)
            self.xsections = self.xsections.append(xsect2)
            self.xsections = self.xsections.append(xsect3)

    def generate_controls(self, frac_open=0, **kwargs):
        controls = []
        if self.control:
            for i, orifice in enumerate(self.orifices['name'].values):
                node1, node2 = orifice.split('_')
                if node1.startswith('ST') and node2.startswith('C'):
                    status = 1
                elif node1.startswith('ST') and node2.startswith('J'):
                    status = 0
                elif node1.startswith('C') and node2.startswith('J'):
                    status = frac_open
                else:
                    raise ValueError()
                rule = ('RULE R{0}\nIF SIMULATION TIME > 0\nTHEN ORIFICE {1} SETTING = {2}\n\n'
                        .format(i, orifice, status))
                controls.append(rule)
        else:
            for i, orifice in enumerate(self.orifices['name'].values):
                node1, node2 = orifice.split('_')
                if node1.startswith('ST') and node2.startswith('C'):
                    status = 0
                elif node1.startswith('ST') and node2.startswith('J'):
                    status = 1
                elif node1.startswith('C') and node2.startswith('J'):
                    status = 0
                else:
                    raise ValueError()
                rule = ('RULE R{0}\nIF SIMULATION TIME > 0\nTHEN ORIFICE {1} SETTING = {2}\n\n'
                        .format(i, orifice, status))
                controls.append(rule)
        self.controls = controls

    def generate_lines(self, transect=True, **kwargs):
        lines = []
        space = None
        justify = None
        lines.append('[TITLE]')
        lines.append('\n')
        lines.append(self.title)
        lines.append('\n\n')
        lines.append('[OPTIONS]')
        lines.append('\n')
        for key, value in self.options.items():
            lines.append("{0}\t{1}\n".format(key, value))
        lines.append('\n')
        lines.append('[EVAPORATION]')
        lines.append('\n')
        for key, value in self.evaporation.items():
            lines.append("{0}\t{1}\n".format(key, value))
        lines.append('\n')
        lines.append('[REPORT]')
        lines.append('\n')
        for key, value in self.report.items():
            lines.append("{0}\t{1}\n".format(key, value))
        lines.append('\n\n')
        lines.append('[RAINGAGES]')
        lines.append('\n')
        lines.append(self.raingages.to_csv(sep='\t', header=None, index=None))
        lines.append('\n\n')
        lines.append('[SUBCATCHMENTS]')
        lines.append('\n')
        lines.append(self.subcatchments.to_csv(sep='\t', header=None, index=None))
        lines.append('\n\n')
        lines.append('[SUBAREAS]')
        lines.append('\n')
        lines.append(self.subareas.to_csv(sep='\t', header=None, index=None))
        lines.append('\n\n')
        lines.append('[INFILTRATION]')
        lines.append('\n')
        lines.append(self.infiltration.to_csv(sep='\t', header=None, index=None))
        lines.append('\n\n')
        lines.append('[JUNCTIONS]')
        lines.append('\n')
        lines.append(self.junctions.to_csv(sep='\t', header=None, index=None))
        lines.append('\n\n')
        lines.append('[STORAGE]')
        lines.append('\n')
        lines.append(self.storage.to_csv(sep='\t', header=None, index=None))
        lines.append('\n\n')
        lines.append('[OUTFALLS]')
        lines.append('\n')
        lines.append(self.outfalls.to_csv(sep='\t', header=None, index=None))
        lines.append('\n\n')
        lines.append('[ORIFICES]')
        lines.append('\n')
        lines.append(self.orifices.to_csv(sep='\t', header=None, index=None))
        lines.append('\n\n')
        lines.append('[CONDUITS]')
        lines.append('\n')
        lines.append(self.conduits.to_csv(sep='\t', header=None, index=None))
        lines.append('\n\n')
        lines.append('[XSECTIONS]')
        lines.append('\n')
        lines.append(self.xsections.to_csv(sep='\t', header=None, index=None))
        lines.append('\n\n')
        if transect:
            lines.append('[TRANSECTS]')
            lines.append('\n')
            lines.append(self.transects)
            lines.append('\n\n')
        lines.append('[TIMESERIES]')
        lines.append('\n')
        lines.append(self.timeseries.to_csv(sep='\t', header=None, index=None))
        lines.append('\n\n')
        lines.append('[CONTROLS]')
        lines.append('\n')
        lines.append(''.join(self.controls))
        lines.append('\n\n')
        lines.append(self.mapconfig)
        lines.append('\n\n')
        lines.append('[COORDINATES]')
        lines.append('\n')
        lines.append(self.coordinates.to_csv(sep='\t', header=None, index=None))
        lines.append('\n\n')
        lines.append('[POLYGONS]')
        lines.append('\n')
        lines.append(self.polygons.to_csv(sep='\t', header=None, index=None))
        self.lines = lines

    def to_file(self, filename, **kwargs):
        with open(filename, 'w') as outfile:
            outfile.writelines(self.lines)

    def run_swmm_ingester(self, out_file, outlet, into_outlet, intensity=1.5, ixes=[], transect=True, position='top', frac_open=0, uniform_n=0, uniform_imperv=0):
        self.generate_title()
        self.generate_options()
        self.generate_evaporation()
        self.generate_report()
        self.generate_raingages()
        self.generate_subcatchments(uniform_imperv=uniform_imperv)
        self.generate_subareas(uniform_n=uniform_n)
        self.generate_infiltration()
        self.generate_junctions()
        self.generate_conduits()
        self.generate_channel_dims()
        self.generate_xsections()
        if transect:
            self.generate_transects()
        self.generate_orifices()
        self.generate_outfalls()
        self.generate_timeseries(intensity=intensity)
        self.generate_storage_uncontrolled(ixes)
        self.generate_storage_controlled(ixes)
        self.generate_storage()
        self.generate_coordinates()
        self.generate_polygons()
        self.generate_map()
        self.generate_control_points(ixes, transect=transect, position=position)
        self.generate_controls(frac_open=frac_open)
        self.generate_lines(transect=transect)
        self.to_file(out_file)
