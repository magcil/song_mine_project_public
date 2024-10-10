import { CheckCircleOutlined, CloseCircleOutlined } from "@ant-design/icons";
import { Progress, Tag, Tooltip } from "antd";

export const deviceColumns = [
  {
    title: "Availablity",
    dataIndex: "is_online",
    key: "is_online",
    render: (text: any, row: any) => {
      return row.is_online ? <CheckCircleOutlined color="green" style={{color:"green"}}/> : <CloseCircleOutlined color="red" style={{color:"red"}}/>;
    },
  },
  {
    title: "Device Name",
    dataIndex: "device_name",
    key: "device_name",
  },
  {
    title: "STATUS",
    key: "status",
    dataIndex: "status",
    render: (text: any) => <Tag color=""> {text}</Tag>,
  },
  {
    title: "Memory Used",
    key: "memory_usage",
    dataIndex: "memory_usage",
    render: (text: any, row: any) => {
      const usage = row.memory_usage;
      const total = row.memory_total;
      const percent = Math.round((usage / total) * 100);
      return (
        <div className="device-memory-usage">
          <Tooltip title={`${usage} / ${total}`}>
            <Progress
              size="small"
              type="circle"
              width={75}
              percent={percent}
              strokeColor={
                percent > 90 ? "red" : percent > 60 ? "orange" : "blue"
              }
            />
          </Tooltip>
        </div>
      );
    },
  },
  {
    title: "Public IP Address",
    key: "public_address",
    dataIndex: "public_address",
  },
  {
    title: "Local IP Address",
    key: "ip_address",
    dataIndex: "ip_address",
  },
  {
    title: "Mac Address",
    key: "mac_address",
    dataIndex: "mac_address",
  },
];


export const detectionColumns = [
  {
    title: "Winner",
    dataIndex: "winner",
    key: "winner",
    
  },
  {
    title: "Score",
    dataIndex: "score",
    key: "score",
  },
  {
    title: "Datetime",
    key: "datetime",
    dataIndex: "datetime",
  },
  {
    title: "Device",
    key: "device",
    dataIndex: "device",
  },
];
