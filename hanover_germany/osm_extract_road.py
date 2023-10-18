from lxml import etree

osm_source = 'osm/hanover_512x512_map.osm'

tree = etree.parse(osm_source)

osm = tree.getroot()

# 查找并删除所有<relation>节点
for relation in osm.findall(".//relation"):
    osm.remove(relation)

# 参考 https://wiki.openstreetmap.org/wiki/Key:highway
# Roads
# ways_with_highway_primary = osm.xpath('//way[tag[@k="highway" and @v="primary"]]')
# ways_with_highway_motorway = osm.xpath('//way[tag[@k="highway" and @v="motorway"]]')
# ways_with_highway_trunk = osm.xpath('//way[tag[@k="highway" and @v="trunk"]]')
# ways_with_highway_secondary = osm.xpath('//way[tag[@k="highway" and @v="secondary"]]')
# ways_with_highway_tertiary = osm.xpath('//way[tag[@k="highway" and @v="tertiary"]]')
# ways_with_highway_unclassified = osm.xpath('//way[tag[@k="highway" and @v="unclassified"]]')
# ways_with_highway_residential = osm.xpath('//way[tag[@k="highway" and @v="residential"]]')

# Roads
valid_highways_v = {
    "primary",
    "motorway",
    "trunk",
    "secondary",
    "tertiary",
    "unclassified",
    "residential"
}

# 有效节点 id
valid_node_ids = set()

# 遍历<way>节点并删除不符合条件的节点
for way in osm.findall(".//way"):
    tags = way.findall("tag[@k='highway']")
    if not any(tag.get("v") in valid_highways_v for tag in tags):
        osm.remove(way)
    else:
        for nd in way.findall("nd"):
            valid_node_ids.add(nd.get('ref'))

# 删除用不到的 node 节点
for node in osm.findall(".//node"):
    if node.get("id") not in valid_node_ids:
        osm.remove(node)

tree.write(r'osm\hanover_512x512_map-roads.osm', encoding='utf-8')
