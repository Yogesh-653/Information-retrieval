create database coventry;
use coventry;
SELECT * from cgl;
select author,count(author) from cgl group by author order by count(author) desc;
select count(distinct(author)) from cgl;
select count(*) from cgl;