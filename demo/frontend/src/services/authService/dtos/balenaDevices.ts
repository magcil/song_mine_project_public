export interface BalenaDevices {
    id: number
    belongs_to__application: BelongsToApplication
    belongs_to__user: any
    actor: number
    should_be_running__release: any
    device_name: string
    is_of__device_type: IsOfDeviceType
    uuid: string
    is_running__release: IsRunningRelease
    note: any
    local_id: any
    status: string
    is_online: boolean
    last_connectivity_event: string
    is_connected_to_vpn: boolean
    last_vpn_event: string
    ip_address: string
    mac_address: string
    public_address: string
    os_version: string
    os_variant: string
    supervisor_version: string
    should_be_managed_by__supervisor_release: any
    should_be_operated_by__release: ShouldBeOperatedByRelease
    is_managed_by__service_instance: any
    provisioning_progress: any
    provisioning_state: string
    download_progress: any
    is_web_accessible: boolean
    longitude: string
    latitude: string
    location: string
    custom_longitude: string
    custom_latitude: string
    is_locked_until__date: any
    is_accessible_by_support_until__date: any
    created_at: string
    modified_at: string
    is_active: boolean
    api_heartbeat_state: string
    memory_usage: number
    memory_total: number
    storage_block_device: string
    storage_usage: number
    storage_total: number
    cpu_temp: number
    cpu_usage: number
    cpu_id: string
    is_undervolted: boolean
    logs_channel: any
    vpn_address: any
  }
  
  export interface BelongsToApplication {
    __id: number
  }
  
  export interface IsOfDeviceType {
    __id: number
  }
  
  export interface IsRunningRelease {
    __id: number
  }
  
  export interface ShouldBeOperatedByRelease {
    __id: number
  }
  