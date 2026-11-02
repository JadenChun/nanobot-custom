//go:build android

package main

import (
	"net"
	"net/netip"
	"time"

	"tailscale.com/net/netmon"
)

// Android denies app UIDs NETLINK_ROUTE access, so net.Interfaces fails.
// Register a synthetic interface derived from the kernel's outbound route;
// this mirrors tailscale.com/feature/androidbin for tsnet c-shared builds.
func init() {
	netmon.RegisterInterfaceGetter(func() ([]netmon.Interface, error) {
		var addrs []net.Addr
		if ip, ok := outboundIP("udp4", "8.8.8.8:53"); ok {
			addrs = append(addrs, &net.IPNet{IP: ip.AsSlice(), Mask: net.CIDRMask(32, 32)})
		}
		if ip, ok := outboundIP("udp6", "[2001:4860:4860::8888]:53"); ok {
			addrs = append(addrs, &net.IPNet{IP: ip.AsSlice(), Mask: net.CIDRMask(128, 128)})
		}
		return []netmon.Interface{{
			Interface: &net.Interface{
				MTU:   1500,
				Name:  "android",
				Flags: net.FlagUp | net.FlagRunning,
			},
			AltAddrs: addrs,
		}}, nil
	})
}

func outboundIP(network, address string) (netip.Addr, bool) {
	dialer := net.Dialer{Timeout: 2 * time.Second}
	conn, err := dialer.Dial(network, address)
	if err != nil {
		return netip.Addr{}, false
	}
	defer conn.Close()
	udp, ok := conn.LocalAddr().(*net.UDPAddr)
	if !ok {
		return netip.Addr{}, false
	}
	ip, ok := netip.AddrFromSlice(udp.IP)
	if !ok {
		return netip.Addr{}, false
	}
	ip = ip.Unmap()
	return ip, !ip.IsLoopback() && !ip.IsUnspecified()
}
