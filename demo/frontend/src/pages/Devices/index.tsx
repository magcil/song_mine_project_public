import { Card, Col, Pagination, Radio, Row, Table } from "antd";
import { observer } from "mobx-react";
import { useStores } from "stores/rootStore";
import { detectionColumns, deviceColumns } from "./columns";
import { useDevices } from "./hooks/useDevices";
import "./index.css";
import detectionService from "services/museekService/detectionService";
import { useDetections } from "./hooks/useDetections";
import DetectionTable from "./components/DetectionTable";
const Devices: React.FC = observer(() => {
  const onChange = (e: any) => console.log(`radio checked:${e.target.value}`);
  const { deviceStore } = useStores();
  useDevices();



  return (
    <>
      <div className="tabled">
        <Row gutter={[24, 0]}>
          <Col xs="24" xl={24}>
            <Card
              bordered={false}
              className="criclebox tablespace mb-24"
              title="Devices"
              extra={
                <>
                  <Radio.Group onChange={onChange} defaultValue="a">
                    <Radio.Button value="a">All</Radio.Button>
                    <Radio.Button value="b">ONLINE</Radio.Button>
                  </Radio.Group>
                </>
              }
            >
              <div className="table-responsive">
                <Table
                  rowKey={(record) => record.id}
                  columns={deviceColumns}
                  dataSource={deviceStore.devices}
                  // bordered
                  className="ant-border-space"
                  onRow={(record, rowIndex) => {
                    return {
                      onClick: (event) => {}, // click row
                      onDoubleClick: (event) => {}, // double click row
                      onContextMenu: (event) => {}, // right button click row
                      onMouseEnter: (event) => {}, // mouse enter row
                      onMouseLeave: (event) => {}, // mouse leave row
                    };
                  }}
                  expandable={{
                    expandedRowRender:  (record) => {
                       
                      return <DetectionTable name = {record.device_name}/> 
                    },
                  }}
                />
              </div>
            </Card>
          </Col>
        </Row>
        {/* <div className="pagination-wrapper">
          <Pagination
            defaultCurrent={1}
            total={50}
            className="pagination-space"
          />
        </div> */}
      </div>
    </>
  );
});

export default Devices;
