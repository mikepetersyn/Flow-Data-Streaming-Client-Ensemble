# command line parameter example: select features from dataset
# parameter pattern: '--feature 1-[0|1|2] 2-1 3-0'

feature_selector = {
  # 0: [-1, -1, -1], # year

  #   +- floats
  #   |   +- bit pattern
  #   |   |   +- one hot
  #   |   |   |
  #   v   v   v
  0: [4, 32, -1],  # src_addr
  1: [4, 32, -1],  # dst_addr
  2: None,  # packets
  3: None,  # bytes
  4: None,  # first switched
  5: None,  # last switched
  6: [1, 16, -1],  # src_port
  7: [1, 16, -1],  # dst_port
  8: [1, 8, -1],  # tcp_flags
  9: [1, 8, -1],  # protocol
  10: None,  # export host
  11: None,  # flow seq number
  12: None,  # duration
  13: None,  # bitrate
  14: [1, 8, 250],  # src_country_code
  15: [1, -1, -1],  # src_longitude
  16: [1, -1, -1],  # src_latitude
  17: [1, 16, -1],  # src_asn
  18: [4, 32, -1],  # src_network
  19: [1, 6, -1],  # src_prefix_len
  20: [1, 12, -1],  # src_vlan
  21: [-1, 1, -1],  # src_locality
  22: [1, 8, 250],  # dst_country_code
  23: [1, -1, -1],  # dst_longitude
  24: [1, -1, -1],  # dst_latitude
  25: [1, 16, -1],  # dst_asn
  26: [4, 32, -1],  # dst_network
  27: [1, 6, -1],  # dst_prefix_len
  28: [1, 12, -1],  # dst_vlan
  29: [-1, 1, -1],  # dst_locality
  30: None,  # year
  31: [1, 4, 12],  # month
  32: [1, 5, 31],  # day
  33: [1, 5, 24],  # hour
  34: [1, 6, 60],  # minute
  35: [1, 6, 60],  # second
  }
