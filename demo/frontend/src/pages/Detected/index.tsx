import { Card, Col, Row, Table } from "antd";
import { observer } from "mobx-react";
import { detectionColumns } from "pages/Devices/columns";
import { useDetections } from "pages/Devices/hooks/useDetections";
import { useStores } from "stores/rootStore";

const Songs: React.FC = observer(() => {
  const { detectionStore } = useStores();
  useDetections(undefined, 1, 15);

  return (
    <>
      <div className="tabled">
        <Row gutter={[24, 0]}>
          <Col xs="24" xl={24}>
            <Card
              bordered={false}
              className="criclebox tablespace mb-24"
              title="Detections"
            >
              <div className="table-responsive">
                <Table
                  columns={detectionColumns}
                  dataSource={detectionStore.paginatedResults?.data || []}
                  className="ant-border-space"
                  pagination={{
                    defaultPageSize: 15,
                    pageSize:15,
                    total: detectionStore.paginatedResults?.totalCount || 0,
                    showSizeChanger: true,
                    onChange: (page, pageSize) => {
                      console.log(page, pageSize);
                      console.log(page, pageSize);
                      detectionStore.getDetectedData(
                        undefined,
                        page,
                        pageSize || 15
                      );
                    },
                  }}
                />
              </div>
            </Card>
          </Col>
        </Row>
      </div>
    </>
  );
});

export default Songs;
