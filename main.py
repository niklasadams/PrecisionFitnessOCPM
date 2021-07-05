# -*- coding: utf-8 -*-
from ocpa.objects.log.importer.mdl import factory as mdl_import_factory
from ocpa.algo.discovery.ocpn import algorithm as ocpn_discovery_factory
from ocpa.visualization.oc_petri_net import factory as pn_vis_factory
import gc
from itertools import combinations
import copy
import pandas as pd
from collections import Counter
import time
import itertools
from itertools import chain
from itertools import product
from ocpa.objects.oc_petri_net.obj import ObjectCentricPetriNet
from matplotlib import pyplot as plt

def enabled(model,state,transition):
        enabled = True
        for a in transition.in_arcs:
            if a.source not in state[0].keys():
                enabled = False
            elif len(list(state[0][a.source].elements())) < 1:
                enabled = False
        return enabled
    
def context_to_string_verbose(context):
    cstr = ""
    for key in context.keys():
        cstr+=key
        for c in context[key].most_common():
            cstr+=",".join(c[0])
            cstr+=str(c[1])
    return cstr

def context_to_string(context):
    return hash(tuple([hash(tuple(sorted(context[cc].items()))) for cc in context.keys()]))
    cstr = ""
    for key in context.keys():
        cstr+=key
        for c in context[key].most_common():
            cstr+=",".join(c[0])
            cstr+=str(c[1])
    return cstr

def patched_most_common(self):
    return sorted(self.items(), key=lambda x: (-x[1],x[0]))


def row_to_binding(row,object_types):
    return (row["event_activity"],{ot:row[ot] for ot in object_types},row["event_id"])

def to_bindings_list(b_dict):
    return [b_dict[e_id] for e_id in sorted(b_dict.keys())]
    
def calculate_context(ocel, object_types):
    print("Constructing contexts...")
    contexts = {}
    bindings = {}
    oh_mappings = {}
    for index, event in ocel.iterrows():
        if index%250 == 0:
             print("event "+str(index))
        binding_executions = {}
        history = {}
        objs = [(ot,o) for ot in object_types for o in event[ot]]
        #find last event for each object
        for (ot,o) in objs:
            mask = ocel[ot].apply(lambda v: True if o in v else False)
            last_index = ocel.where((mask) & (index > ocel["event_id"])).last_valid_index()
            if last_index:
                history = {k: max(history[k], oh_mappings[last_index][k]) if k in history and k in oh_mappings[last_index] else history.get(k, oh_mappings[last_index].get(k)) for k in history.keys() | oh_mappings[last_index].keys()}
        for (ot,o) in objs:
            history[(ot, o)] = index
        oh_mappings[index] = history
        context = {}
        for ot in object_types:
            context[ot] = Counter()
        for (ot, o) in history.keys():
            mask = ocel[ot].apply(lambda v: True if o in v else False)
            prefix =  tuple(["START"] + ocel.where((mask) & (history[(ot,o)] >= ocel["event_id"]) & (index > ocel["event_id"] ))["event_activity"].dropna().to_list())
            binding_executions.update({binding[2]: (binding[0],binding[1]) for binding in (ocel.where((mask) & (history[(ot,o)] >= ocel["event_id"]) & (index > ocel["event_id"] ))[["event_id","event_activity"]+object_types].dropna().apply(lambda x: row_to_binding(x,object_types) if len(x) != 0 else [],axis = 1))})
            context[ot] += Counter([prefix])
        contexts[event["event_id"]]= context
        bindings[event["event_id"]]= to_bindings_list(binding_executions)
    return contexts, bindings

def calculate_en_l(ocel, contexts, object_types):
    print("Calculating en_l ...")
    context_mapping = {}
    log_contexts = {}
    for index, event in ocel.iterrows():
        context = context_to_string(contexts[event["event_id"]])
        if context not in log_contexts.keys():
            log_contexts[context] = [index]
            context_mapping[context] = contexts[event["event_id"]]
        else:
            log_contexts[context].append(index)
    en_l = {}
    for context in log_contexts.keys():
        event_ids = log_contexts[context]
        en_l[context] = []
        #for each event, for each object type see the following activity
        for e in event_ids:
            en_l[context].append(ocel.at[e,"event_activity"])
        en_l[context] = list(set(en_l[context]))           
    return en_l

def model_enabled(model,state,transition):
        enabled = True
        for a in transition.in_arcs:
            if a.source not in state.keys():
                enabled = False
            elif len(state[a.source]) < 1:
                enabled = False
        return enabled

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1,len(s)+1))

def combinations_without_repetitions(l,k):
    results = []
    for i in range(1,k+1):
        results+= list(itertools.combinations(l,i))
    return list(set(results))
    
def convertToNumber(s):
    return int.from_bytes(s.encode(), 'little')
def convert_place_to_number(p):
    return convertToNumber(p.name)
def convert_object_to_number(o):
    return convertToNumber(o[0]+str(o[1]))
def place_compare(p1,p2):
    if convertToNumber(p1.name) > convertToNumber(p2.name):
        return 1
    elif convertToNumber(p1.name) < convertToNumber(p2.name):
        return -1
    else:
        return 0

def state_to_string(state):
    s = ""
    k = sorted(list(state.keys()),key = convert_place_to_number)
    for key in k:
        s+=key.name
        s+=",".join([c[0]+","+str(c[1]) for c in state[key]])
    return s

def no_prefix_loop(prefixes):
    return not any([len(p) != len(set(p)) for p in prefixes.values()])

def state_to_place_counter(state):
    result = Counter()
    for p in state.keys():
        result+= Counter({p.name:len(state[p])})
    return hash(tuple(sorted(result.items())))

def binding_possible(ocpn,state,binding,object_types):
    if len(binding) == 0:
        return False
    tokens = [o for ot in object_types for o in binding[0][1][ot]]
    input_places_tokens = []
    t = None
    for t_ in ocpn.transitions:
        if t_.name == binding[0][0]:
            t = t_
    if t == None:
        return False
    for a in t.in_arcs:
        input_places_tokens += state[a.source] if a.source in state.keys() else []
    if set(tokens).issubset(set(input_places_tokens)) or set(tokens) == set(input_places_tokens):
        if model_enabled(ocpn,state,t):
            return True
    return False

def calculate_next_states_on_bindings(ocpn, state, binding, object_types):
    state_update_pairs = []
    
    if binding_possible(ocpn,state,binding,object_types):
        t = None
        for t_ in ocpn.transitions:
            if t_.name == binding[0][0]:
                t = t_
        in_places = {}
        out_places = {}
        for ot in object_types:
            in_places[ot] = [(x,y) for (x,y) in [(a.source,a.variable) for a in t.in_arcs] if x.object_type == ot]
            out_places[ot] = [x for x in [a.target for a in t.out_arcs] if x.object_type == ot]
        new_state = {k:state[k].copy() for k in state.keys()}
        update = not t.silent
        for ot in object_types:
            if ot not in binding[0][1].keys():
                continue
            for out_pl in out_places[ot]:
                if out_pl not in new_state.keys():
                    new_state[out_pl] = []
                new_state[out_pl]+=list(binding[0][1][ot])
            
            for (in_pl,is_v) in in_places[ot]:
                new_state[in_pl] = list((Counter(new_state[in_pl]) - Counter(list(binding[0][1][ot]))).elements())
                if new_state[in_pl] == []:
                    del new_state[in_pl]
            state_update_pairs.append((new_state,update))
        
    else:
        for t in ocpn.transitions:
            if t.silent:
                if model_enabled(ocpn,state,t):
                    input_tokens = {ot:[] for ot in object_types}
                    input_token_combinations = {ot:[] for ot in object_types}
                    in_places = {}
                    out_places = {}
                    for ot in object_types:
                        in_places[ot] = [(x,y) for (x,y) in [(a.source,a.variable) for a in t.in_arcs] if x.object_type == ot]
                        out_places[ot] = [x for x in [a.target for a in t.out_arcs] if x.object_type == ot]
                        token_lists = [[z for z in state[x]] for (x,y) in in_places[ot]]
                        #is_variable = any(y for (x,y) in in_places[ot])
                        if len(token_lists) != 0:
                            input_tokens[ot] = set.intersection(*map(set,token_lists))
                            input_token_combinations[ot] = list(combinations(input_tokens[ot], 1)) #if not is_variable else list(powerset_combinations(input_tokens[ot],prefixes)))#list(powerset_combinations(input_tokens[ot],prefixes)))
                        else:
                            input_tokens[ot] = set()
                    indices_list = [list(range(len(input_token_combinations[ot]))) if len(input_token_combinations[ot]) != 0 else [-1] for ot in object_types]
                    possible_combinations = list(product(*indices_list))
                    for comb in possible_combinations:
                        binding_silent = {}
                        for i in range(len(object_types)):
                            ot = object_types[i]
                            if -1 == comb[i]:
                                continue
                            binding_silent[ot] = input_token_combinations[ot][comb[i]]
                        new_state = {k:state[k].copy() for k in state.keys()}
                        update = not t.silent
                        for ot in object_types:
                            if ot not in binding_silent.keys():
                                continue
                            for out_pl in out_places[ot]:
                                if out_pl not in new_state.keys():
                                    new_state[out_pl] = []
                                new_state[out_pl]+=list(binding_silent[ot])
                            for (in_pl,is_v) in in_places[ot]:
                                new_state[in_pl] = list((Counter(new_state[in_pl]) - Counter(list(binding_silent[ot]))).elements())
                                if new_state[in_pl] == []:
                                    del new_state[in_pl]
                        state_update_pairs.append((new_state,update))
    return state_update_pairs

def update_binding(binding,update):
    if update:
        return copy.deepcopy(binding[1:])
    else:
        return copy.deepcopy(binding)

def calculate_en_m(contexts,bindings,ocpn,object_types):
    print("Calculating en_m ....")
    results = {}
    times = [0,0,0,0,0,0]
    for i in range(0,len(contexts)):
        if i%250 == 0:
            print("event "+str(i))
        context = contexts[i] 
        binding = bindings[i]
        q = []
        state_binding_set = set()
        initial_node = [{},binding]
        all_objects = {}
        for ot in object_types:
            all_objects[ot] = set()
            for b in binding:
                for o in b[1][ot]:
                    all_objects[ot].add((ot,o))
        for color in context.keys():
            
            tokens = all_objects[color]
            #if tokens ar enot in the bindings but the context indicates that they should be in the inital place they need to be added
            to_be_added = 0
            if len(tokens) != len(list(context[color].elements())):
                to_be_added = len(list(context[color].elements())) - len(tokens)
               
            if tokens == set() and to_be_added == 0:
                continue
            for p in ocpn.places:
                if p.object_type == color and p.initial:
                    #add START Tokens for each prefix a new token with new id
                    initial_node[0][p] = []
                    for (ot,o) in tokens:
                        initial_node[0][p].append((ot,o))
                    for i in range(0,to_be_added):
                        initial_node[0][p].append((ot,"additional"+str(i)))
                        
        initial_node = [initial_node]
        #transform bindings such that objects are identified as ot o not only o
        for b in binding:
            for ot in object_types:
                b[1][ot] = [(ot,o) for o in b[1][ot]]
        [q.append(node)for node in initial_node]
        index = 0
        context_string_target = context_to_string(context)
        if context_string_target not in results.keys():
            results[context_string_target] = set()
        [state_binding_set.add((state_to_place_counter(elem[0]),len(elem[1]))) for elem in q]
        while not index == len(q):
            elem = q[index]
            index+=1
            if len(elem[1]) == 0:
                for t in ocpn.transitions:
                    if model_enabled(ocpn,elem[0],t) and not t.silent:
                        results[context_string_target].add(t.name)
            t = time.time()  
            state_update_pairs = calculate_next_states_on_bindings(ocpn, elem[0], elem[1], object_types)
            times[1]+= time.time()-t 
            #for all next states
            for (state, update) in state_update_pairs:
                t = time.time() 
                updated_binding = update_binding(elem[1],update)
                times[3]+= time.time()-t 
                t = time.time()
                traditional_state_string = state_to_place_counter(state)
                times[2]+= time.time()-t 
                t = time.time() 
                if (traditional_state_string,len(updated_binding)) in state_binding_set:
                    continue
                state_binding_set.add((traditional_state_string,len(updated_binding)))
                q.append([state,updated_binding])
        del q   
        gc.collect()     
    return results


def calculate_precision(ocel,context_mapping,en_l,en_m):
    prec = []
    fit = []
    skipped = 0
    for index, row in ocel.iterrows():
        e_id = row["event_id"]
        context = context_mapping[e_id]
        en_l_a = en_l[context_to_string(context)]
        en_m_a = en_m[context_to_string(context)]
        if len(en_m[context_to_string(context)]) == 0 or (set(en_l_a).intersection(en_m_a) == set()):
            skipped+=1
            fit.append(0)
            continue
        prec.append(len(set(en_l[context_to_string(context)]).intersection(set(en_m[context_to_string(context)])))/float(len(en_m[context_to_string(context)])))
        fit.append(len(set(en_l[context_to_string(context)]).intersection(set(en_m[context_to_string(context)])))/float(len(en_l[context_to_string(context)])))
    return sum(prec)/len(prec), skipped, sum(fit)/len(fit)
    
def create_flower_model(ocpn,ots):
    arcs = []
    transitions = []
    places = []
    [places.append(ObjectCentricPetriNet.Place(name=c+"1",object_type=c,initial=True)) for c in ots]
    #[places.append(ObjectCentricPetriNet.Place(name=c+"final",object_type=c,final=True)) for c in ots]
    for t in [x for x in ocpn.transitions if not x.silent]:
        t_new = ObjectCentricPetriNet.Transition(name=t.name)
        transitions.append(t_new)
        for ot in ots:
            if ot in [a.source.object_type for a in t.in_arcs]:
                var = any([a.variable for a in t.in_arcs if a.source.object_type == ot ])
                source_place = [p for p in places if p.initial and p.object_type == ot][0]
                in_a = ObjectCentricPetriNet.Arc(source_place, t_new, variable = var)
                out_a = ObjectCentricPetriNet.Arc(t_new, source_place, variable = var)
                arcs.append(in_a)
                arcs.append(out_a)
                t_new.in_arcs.add(in_a)
                t_new.out_arcs.add(out_a)
    flower_ocpn = ObjectCentricPetriNet(places = places, transitions = transitions, arcs = arcs)
    return flower_ocpn
def create_underfit():
    places = []
    transitions = []
    arcs = []
    item1 = ObjectCentricPetriNet.Place(name="item1",object_type="item",initial = True)
    places.append(item1)
    order1 = ObjectCentricPetriNet.Place(name="order1",object_type="order",initial = True)
    places.append(order1)
    item2 = ObjectCentricPetriNet.Place(name="item2",object_type="item")
    places.append(item2)
    order2 = ObjectCentricPetriNet.Place(name="order2",object_type="order")
    places.append(order2)
    item3 = ObjectCentricPetriNet.Place(name="item3",object_type="item")
    places.append(item3)
    item4 = ObjectCentricPetriNet.Place(name="item4",object_type="item")
    places.append(item4)
    item5 = ObjectCentricPetriNet.Place(name="item5",object_type="item")
    places.append(item5)
    item6 = ObjectCentricPetriNet.Place(name="item6",object_type="item")
    places.append(item6)
    item7 = ObjectCentricPetriNet.Place(name="item7",object_type="item")
    places.append(item7)
    order3 = ObjectCentricPetriNet.Place(name="order3",object_type="order")
    places.append(order3)
    order4 = ObjectCentricPetriNet.Place(name="order4",object_type="order")
    places.append(order4)
    order5 = ObjectCentricPetriNet.Place(name="order5",object_type="order")
    places.append(order5)
    delivery1 = ObjectCentricPetriNet.Place(name="delivery1",object_type="delivery")
    places.append(delivery1)
    delivery2 = ObjectCentricPetriNet.Place(name="delivery2",object_type="delivery")
    places.append(delivery2)
    delivery3 = ObjectCentricPetriNet.Place(name="delivery3",object_type="delivery")
    places.append(delivery3)
    delivery4 = ObjectCentricPetriNet.Place(name="delivery4",object_type="delivery")
    places.append(delivery4)
    delivery5 = ObjectCentricPetriNet.Place(name="delivery5",object_type="delivery")
    places.append(delivery5)
    
    po = ObjectCentricPetriNet.Transition(name="Place Order")
    transitions.append(po)
    co = ObjectCentricPetriNet.Transition(name="Confirm Order")
    transitions.append(co)
    pi = ObjectCentricPetriNet.Transition(name="Pick Item")
    transitions.append(pi)
    lc = ObjectCentricPetriNet.Transition(name="Load Cargo")
    transitions.append(lc)
    fc = ObjectCentricPetriNet.Transition(name="Fuel Car")
    transitions.append(fc)
    sr = ObjectCentricPetriNet.Transition(name="Start Route")
    transitions.append(sr)
    er = ObjectCentricPetriNet.Transition(name="End Route")
    transitions.append(er)
    pr = ObjectCentricPetriNet.Transition(name="Payment Reminder")
    transitions.append(pr)
    p = ObjectCentricPetriNet.Transition(name="Pay Order")
    transitions.append(p)
    
    a1 = ObjectCentricPetriNet.Arc(order1,po)
    po.in_arcs.add(a1)
    arcs.append(a1)
    a2 = ObjectCentricPetriNet.Arc(item1,po,variable=True)
    po.in_arcs.add(a2)
    arcs.append(a2)
    a3 = ObjectCentricPetriNet.Arc(po,order2)
    po.out_arcs.add(a3)
    arcs.append(a3)
    a4 = ObjectCentricPetriNet.Arc(po,item2,variable=True)
    po.out_arcs.add(a4)
    arcs.append(a4)
    a5 = ObjectCentricPetriNet.Arc(order2,co)
    co.in_arcs.add(a5)
    arcs.append(a5)
    a6 = ObjectCentricPetriNet.Arc(item2,co,variable=True)
    co.in_arcs.add(a6)
    arcs.append(a6)
    a7 = ObjectCentricPetriNet.Arc(co,order3)
    co.out_arcs.add(a7)
    arcs.append(a7)
    a8 = ObjectCentricPetriNet.Arc(co,item3,variable=True)
    co.out_arcs.add(a8)
    arcs.append(a8)
    a9 = ObjectCentricPetriNet.Arc(item3,pi)
    pi.in_arcs.add(a9)
    arcs.append(a9)
    a10 = ObjectCentricPetriNet.Arc(pi,item4)
    pi.out_arcs.add(a10)
    arcs.append(a10)
    a11 = ObjectCentricPetriNet.Arc(order3,pr)
    pr.in_arcs.add(a11)
    arcs.append(a11)
    a12 = ObjectCentricPetriNet.Arc(pr,order4)
    pr.out_arcs.add(a12)
    arcs.append(a12)
    a13 = ObjectCentricPetriNet.Arc(order4,p)
    p.in_arcs.add(a13)
    arcs.append(a13)
    a14 = ObjectCentricPetriNet.Arc(p,order5)
    p.out_arcs.add(a14)
    arcs.append(a14)
    a15 = ObjectCentricPetriNet.Arc(delivery1,fc)
    fc.in_arcs.add(a15)
    arcs.append(a15)
    a16 = ObjectCentricPetriNet.Arc(fc,delivery2)
    fc.out_arcs.add(a16)
    arcs.append(a16)
    a17 = ObjectCentricPetriNet.Arc(delivery2,lc)
    lc.in_arcs.add(a17)
    arcs.append(a17)
    a18 = ObjectCentricPetriNet.Arc(item4,lc,variable = True)
    lc.in_arcs.add(a18)
    arcs.append(a18)
    a19 = ObjectCentricPetriNet.Arc(lc,item5,variable = True)
    lc.out_arcs.add(a19)
    arcs.append(a19)
    a20 = ObjectCentricPetriNet.Arc(lc,delivery3)
    lc.out_arcs.add(a20)
    arcs.append(a20)
    a21 = ObjectCentricPetriNet.Arc(delivery3,sr)
    sr.in_arcs.add(a21)
    arcs.append(a21)
    a22 = ObjectCentricPetriNet.Arc(item5,sr,variable = True)
    sr.in_arcs.add(a22)
    arcs.append(a22)
    a23 = ObjectCentricPetriNet.Arc(sr,item6,variable = True)
    sr.out_arcs.add(a23)
    arcs.append(a23)
    a24 = ObjectCentricPetriNet.Arc(sr,delivery4)
    sr.out_arcs.add(a24)
    arcs.append(a24)
    a25 = ObjectCentricPetriNet.Arc(delivery4,er)
    er.in_arcs.add(a25)
    arcs.append(a25)
    a26 = ObjectCentricPetriNet.Arc(item6,er,variable = True)
    er.in_arcs.add(a26)
    arcs.append(a26)
    a27 = ObjectCentricPetriNet.Arc(er,item7,variable = True)
    er.out_arcs.add(a27)
    arcs.append(a27)
    a28 = ObjectCentricPetriNet.Arc(er,delivery5)
    er.out_arcs.add(a28)
    arcs.append(a28)

    underfit = ObjectCentricPetriNet(places = places, transitions = transitions, arcs = arcs)
    return underfit

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
def main():   
    filename = "example.csv"
    ots = ["order","item","delivery"]
    event_df = pd.read_csv(filename,sep=';')
    for ot in ots:
        event_df[ot] = event_df[ot].map(lambda x: [y.strip() for y in x.split(',')] if isinstance(x,str) else [])
    print(event_df)

    event_df["event_id"] = list(range(0,len(event_df)))
    event_df.index = list(range(0,len(event_df)))
    event_df["event_id"] = event_df["event_id"].astype(float).astype(int)
   
    mdl_df = mdl_import_factory.apply(event_df)

    ocpn = ocpn_discovery_factory.apply(mdl_df)
    gviz = pn_vis_factory.apply(ocpn, variant="control_flow", parameters={"format": "svg"})
    pn_vis_factory.save(gviz,"object_model_mined.svg")
    t0 = time.time()
    contexts, bindings = calculate_context(event_df, ots)
    flower_ocpn = create_flower_model(ocpn,ots)
    gviz = pn_vis_factory.apply(flower_ocpn, variant="control_flow", parameters={"format": "svg"})
    pn_vis_factory.save(gviz,"object_model_flower.svg")
    underfit = create_underfit()
    gviz = pn_vis_factory.apply(underfit, variant="control_flow", parameters={"format": "svg"})
    pn_vis_factory.save(gviz,"object_model_restricted.svg")
    good_ocpn = None
    transs = [t for t in ocpn.transitions if t.name == "Item out of stock"][0]
    transr = [t for t in ocpn.transitions if t.name == "Reorder Item"][0]
    item1trans = [a.source.name for a in transs.in_arcs][0]
    item2trans = [a.target.name for a in transs.out_arcs][0]
    item9trans = [a.target.name for a in transr.out_arcs][0]
    for t in ocpn.transitions:
        if t.silent and item1trans in [x.source.name.strip() for x in t.in_arcs]:
            #remove transition and add new
            p_1_in = [x.source for x in t.in_arcs if "item" in x.source.name][0]
            p_2_in = [x.source for x in t.in_arcs if "order" in x.source.name][0]
            p_2_out = [x.target for x in t.out_arcs if "order" in x.target.name][0]
            t_1 = ObjectCentricPetriNet.Transition(name = "t1s",silent = True)
            t_2 = ObjectCentricPetriNet.Transition(name = "t2s",silent = True)
            a_in_1 = ObjectCentricPetriNet.Arc(p_1_in,t_1)
            a_in_2 = ObjectCentricPetriNet.Arc(p_2_in,t_2)
            a_out_2 = ObjectCentricPetriNet.Arc(t_2,p_2_out)
            t_1.in_arcs.add(a_in_1)
            t_2.in_arcs.add(a_in_2)
            t_2.out_arcs.add(a_out_2)
            
            pl3 = [p for p in ocpn.places if p.name == item2trans][0]
            a3in = [a for a in pl3.in_arcs if a.source.silent][0]
            a3out = [a for a in pl3.out_arcs if a.target.silent][0]
            t11 = a3out.target
            at11_out = [a for a in t11.out_arcs][0]
            #pl3.in_arcs.remove(a3in)
            pl3.out_arcs.remove(a3out)
            t11.in_arcs.remove(a3out)
            t11.out_arcs.remove(at11_out)
    
            pl6 = [p for p in ocpn.places if p.name == item9trans][0]
            add_a = ObjectCentricPetriNet.Arc(t_1,pl6)
            t_1.out_arcs.add(add_a)
            good_ocpn = ObjectCentricPetriNet(places = ocpn.places, transitions = [t_ for t_ in ocpn.transitions+[t_1,t_2]  if t_!=t and t_!=t11], arcs = [a  for a in ocpn.arcs+[a_in_1,a_in_2,a_out_2,add_a] if a not in t.in_arcs and a not in t.out_arcs and a != a3in and a!= a3out and a != at11_out])
    
    gviz = pn_vis_factory.apply(good_ocpn, variant="control_flow", parameters={"format": "svg"})
    pn_vis_factory.save(gviz,"object_model_appropriate.svg")
    total = time.time()-t0
    #print(contexts)
    print("Calculating contexts")
    print(total)
    t1 = time.time()
    en_l = calculate_en_l(event_df, contexts, ots)
    total = time.time()-t1
    print("Calculated en_l")
    print(total)
    #print(en_l)
    t2 = time.time()
    print("Flower Model")
    en_m = calculate_en_m(list(contexts.values()),bindings, flower_ocpn,ots)
    total = time.time() - t2
    print("Calculated en_m")
    print(total)
    #print(en_m)
    precision, skipped, fitness = calculate_precision(event_df,contexts,en_l,en_m)
    print("Precision")
    print(precision)
    print("Skipped Events")
    print(skipped)
    print("Skipping Fitness")
    print(str(1 - skipped/len(event_df)))
    print("Fitness")
    print(str(fitness))
    
    print("Appropriate model")
    t2 = time.time()
    en_m = calculate_en_m(list(contexts.values()),bindings, good_ocpn,ots)
    total = time.time() - t2
    print("Calculated en_m")
    print(total)
    precision, skipped, fitness = calculate_precision(event_df,contexts,en_l,en_m)
    print("Precision")
    print(precision)
    print("Skipped Events")
    print(skipped)
    print("Skipping Fitness")
    print(str(1 - skipped/len(event_df)))
    print("Fitness")
    print(str(fitness))
    
    print("Overfitting to single variant model")
    t2 = time.time()
    en_m = calculate_en_m(list(contexts.values()),bindings, underfit,ots)
    total = time.time() - t2
    print("Calculated en_m")
    print(total)
    precision, skipped, fitness = calculate_precision(event_df,contexts,en_l,en_m)
    print("Precision")
    print(precision)
    print("Skipped Events")
    print(skipped)
    print("Skipping Fitness")
    print(str(1 - skipped/len(event_df)))
    print("Fitness")
    print(str(fitness))
    return

if __name__ == "__main__":
    Counter.most_common = patched_most_common
    main()